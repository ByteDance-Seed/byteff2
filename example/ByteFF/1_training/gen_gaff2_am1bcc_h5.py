# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Output schema (matches `byteff_pretrain.yaml`):
    sigma                  (n_atoms, 1)         Å
    epsilon                (n_atoms, 1)         kcal/mol
    bond_k                 (n_bonds, 1)         kcal/mol/Å²
    bond_length            (n_bonds, 1)         Å
    angle_k                (n_angles, 1)        kcal/mol/rad²
    angle_theta            (n_angles, 1)        deg
    propertorsion_k        (n_propers, 4)       kcal/mol  (period=1..4)
    impropertorsion_k      (n_impropers, 1)     kcal/mol  (3 permutations summed)
    partial_charges        (n_atoms, 1)         e         AM1BCC
    coords                 (1, n_atoms, 3)      Å         (copy from src h5)
"""

import argparse
from collections import defaultdict
import logging
from operator import itemgetter
import os
import shutil
import subprocess
import tempfile
import traceback

import h5py
import numpy as np
import pandas as pd

from byteff2.bytemol.core import Conformer, Molecule, MoleculeGraph
from byteff2.bytemol.core.rkutil import sorted_atomids
from byteff2.bytemol.toolkit.gmxtool.topparse import BondTypeEnum, DihedralTypeEnum, TopoAtomTypes, TopoFullSystem
from byteff2.bytemol.units import simple_unit as unit
from byteff2.bytemol.utils import setup_default_logging, temporary_cd


PROPERTORSION_TERMS = 4
logger = setup_default_logging()
logger.setLevel(logging.INFO)


def _run(cmd: str, env_extra: dict = None):
    """Run a shell command, raise on non-zero exit."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


# ---------------------------------------------------------------------------
# AM1BCC charges via antechamber (inlined to avoid byteff dependency)
# ---------------------------------------------------------------------------
def _call_antechamber_am1bcc(
    sdf_file: str, natoms: int, total_formal_charge: int, working_path: str, sqm_opt: bool = False
) -> list:
    """Run antechamber AM1-BCC on an .sdf file and return per-atom partial charges."""
    env_extra = {"AMBER_BCCDAT": "BCCPARM.DAT.am1bcc"}
    sdf_file = os.path.abspath(sdf_file)
    if sqm_opt:
        sqm_params = ["-ek", "\"qm_theory='AM1', scfconv=1.d-10, grms_tol=0.0005, ndiis_attempts=700\" "]
    else:
        sqm_params = [
            "-ek",
            "\"qm_theory='AM1', scfconv=1.d-10, maxcyc=0, grms_tol=0.0005, ndiis_attempts=700\" ",
        ]
    with temporary_cd(working_path):
        cmd = " ".join(
            [
                "antechamber",
                "-i",
                f'"{sdf_file}"',
                "-fi",
                "sdf",
                "-o",
                "charged.mol2",
                "-fo",
                "mol2",
                "-pf",
                "yes",
                "-dr",
                "n",
                "-c",
                "bcc",
                "-nc",
                str(int(total_formal_charge)),
            ]
            + sqm_params
        )
        _run(cmd, env_extra=env_extra)
        cmd = "antechamber -dr n -i charged.mol2 -fi mol2 -o charges2.mol2 -fo mol2 -c wc -cf charges.txt -pf yes"
        _run(cmd, env_extra=env_extra)
        with open("charges.txt", "r") as f:
            partial_charges = [float(c) for c in f.read().split()]
    assert len(partial_charges) == natoms, f"got {len(partial_charges)} charges for {natoms} atoms"
    return partial_charges


def assign_am1bcc_charges(mol: Molecule, work_dir: str, conf_id: int = 0) -> list:
    """Run AM1-BCC on a Molecule's given conformer; charges are normalized to
    sum to the total formal charge."""
    total_formal = int(sum(mol.formal_charges))
    os.makedirs(work_dir, exist_ok=True)
    with temporary_cd(work_dir):
        mol.to_sdf("mol.sdf", conf_id=conf_id)
        partial_charges = _call_antechamber_am1bcc(
            "mol.sdf",
            mol.natoms,
            total_formal,
            working_path="./",
            sqm_opt=False,
        )
    # fix rounding error: distribute residual to all atoms so the sum matches
    deviate = sum(partial_charges) - total_formal
    shift = deviate / len(partial_charges)
    return [c - shift for c in partial_charges]


# ---------------------------------------------------------------------------
# GAFF2 topology via acpype (inlined to avoid byteff dependency)
# ---------------------------------------------------------------------------
def run_gaff2_acpype(mol: Molecule, work_dir: str, mol_name: str, conf_id: int = 0) -> TopoFullSystem:
    """Build a GAFF2 GROMACS itp via acpype and return the parsed TopoFullSystem."""
    os.makedirs(work_dir, exist_ok=True)
    with temporary_cd(work_dir):
        sdf_path = mol_name + ".sdf"
        mol.to_sdf(sdf_path, conf_id=conf_id)
        mol2_path = mol_name + ".mol2"
        _run(f"obabel {sdf_path} -O {mol2_path} -xl")
        total_formal = int(sum(mol.formal_charges))
        _run(f"acpype -i {mol2_path} -c user -a gaff2 -o gmx -b {mol_name} -f -n {total_formal}")
        tfs = TopoFullSystem.from_file(f"{mol_name}.acpype/{mol_name}_GMX.itp")
    return tfs


# ---------------------------------------------------------------------------
# Improper helpers
# ---------------------------------------------------------------------------
def get_impropers_canonical(mol: Molecule):
    """Return a sorted list of canonical impropers (center, j, k, l) where j<k<l.

    Definition matches byteff2.data.data.get_impropers: any sp2-ish C/N atom
    (atomic_number 6 or 7) with exactly 3 neighbors.
    """
    rkmol = mol.get_rkmol()
    atomsets = set()
    for atom in rkmol.GetAtoms():
        if atom.GetAtomicNum() in (6, 7):
            nei = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nei) == 3:
                atomsets.add(sorted_atomids((atom.GetIdx(), nei[0], nei[1], nei[2]), is_improper=True))
    out = sorted(atomsets, key=itemgetter(0, 1, 2, 3))
    return out


# ---------------------------------------------------------------------------
# Extract numerical params from a TopoFullSystem produced by acpype -a gaff2
# ---------------------------------------------------------------------------
def gaff2_params_to_arrays(tfs: TopoFullSystem, mol: Molecule) -> dict:
    """Convert a parsed GAFF2 itp (TopoFullSystem) into numpy arrays
    aligned with MoleculeGraph(mol) topology ordering.
    """
    itp_mol = tfs.mol_topos[0]
    natoms = len(itp_mol.atoms)
    assert natoms == mol.natoms, f"natoms mismatch: itp={natoms} mol={mol.natoms}"

    # ------- per-atom sigma/epsilon (look up by atype) -------
    atom_types = TopoAtomTypes(tfs.uuid)
    type_to_se = {}
    for at in atom_types.atomtypes:
        # at.sigma / at.epsilon are in nm / kJ
        type_to_se[at.name] = (unit.nm_to_A(at.sigma), unit.kj_to_kcal(at.epsilon))
    sigma = np.zeros((natoms, 1), dtype=np.float64)
    epsilon = np.zeros((natoms, 1), dtype=np.float64)
    for i, atom in enumerate(itp_mol.atoms):
        s, e = type_to_se[atom.atype]
        sigma[i, 0] = s
        epsilon[i, 0] = e

    # ------- bonds: build dict (i,j)->(length,k) -------
    bond_d = {}
    for bond in itp_mol.bonds:
        assert bond.funct == BondTypeEnum.BOND
        idxs = sorted_atomids((bond.ai - 1, bond.aj - 1))
        # bond.c0 (nm) -> Å, bond.c1 (kJ/mol/nm²) -> kcal/mol/Å²
        bond_d[idxs] = (unit.nm_to_A(bond.c0), unit.kj_mol_nm2_to_kcal_mol_A2(bond.c1))

    # ------- angles -------
    angle_d = {}
    for ang in itp_mol.angles:
        idxs = sorted_atomids((ang.ai - 1, ang.aj - 1, ang.ak - 1))
        # ang.c0 (deg), ang.c1 (kJ/mol/rad²) -> kcal/mol/rad²
        angle_d[idxs] = (ang.c0, unit.kJ_mol_to_kcal_mol(ang.c1))

    # ------- dihedrals: split propers (MULTIPLE_PROPER) and impropers (PERIODIC_IMPROPER) -------
    # Convention used by mbis_bonded_*.h5 / byteff2: period n has phase = 180° if
    # n is even else 0°.  acpype/GAFF2 may emit a phase that disagrees with this
    # convention; in that case we flip the sign of k so that the physical
    # potential V = k * (1 + cos(n*phi - phase)) is preserved.
    proper_terms = defaultdict(list)  # idxs -> list of (period, k_kcal)
    improper_terms = defaultdict(list)  # idxs -> list of (period, k_kcal)
    for d in itp_mol.dihedrals:
        idxs = (d.ai - 1, d.aj - 1, d.ak - 1, d.al - 1)
        period = int(round(d.c2))
        k_kcal = unit.kj_to_kcal(d.c1)
        canonical_phase = 180.0 if period % 2 == 0 else 0.0
        # If the itp phase differs from the convention by 180°, flip the sign of k.
        if abs(((d.c0 - canonical_phase) % 360.0) - 180.0) < 1.0:
            k_kcal = -k_kcal
        if d.funct == DihedralTypeEnum.MULTIPLE_PROPER:
            sidx = sorted_atomids(idxs)
            proper_terms[sidx].append((period, k_kcal))
        elif d.funct == DihedralTypeEnum.PERIODIC_IMPROPER:
            improper_terms[idxs].append((period, k_kcal))
        else:
            raise ValueError(f"unsupported dihedral funct {d.funct}")

    # ------- align to MoleculeGraph topology -------
    mg = MoleculeGraph(mol)
    intra = mg.get_intra_topo()
    bonds_canon = [tuple(b) for b in intra["Bond"]]
    angles_canon = [tuple(a) for a in intra["Angle"]]
    propers_canon = [tuple(p) for p in intra["ProperTorsion"]]
    impropers_canon = get_impropers_canonical(mol)

    # bonds
    bond_k = np.zeros((len(bonds_canon), 1), dtype=np.float64)
    bond_length = np.zeros((len(bonds_canon), 1), dtype=np.float64)
    for i, b in enumerate(bonds_canon):
        sb = sorted_atomids(b)
        if sb not in bond_d:
            raise KeyError(f"bond {sb} missing in itp for {mol.name}")
        l, k = bond_d[sb]
        bond_length[i, 0] = l
        bond_k[i, 0] = k

    # angles
    angle_k = np.zeros((len(angles_canon), 1), dtype=np.float64)
    angle_theta = np.zeros((len(angles_canon), 1), dtype=np.float64)
    for i, a in enumerate(angles_canon):
        sa = sorted_atomids(a)
        if sa not in angle_d:
            raise KeyError(f"angle {sa} missing in itp for {mol.name}")
        t, k = angle_d[sa]
        angle_theta[i, 0] = t
        angle_k[i, 0] = k

    # propers (n_propers, 4): each column = period (1..4)
    propertorsion_k = np.zeros((len(propers_canon), PROPERTORSION_TERMS), dtype=np.float64)
    for i, p in enumerate(propers_canon):
        sp = sorted_atomids(p)
        terms = proper_terms.get(sp, [])
        for period, k_kcal in terms:
            if 1 <= period <= PROPERTORSION_TERMS:
                propertorsion_k[i, period - 1] += k_kcal

    # impropers (n_impropers, 1): each acpype itp improper line has 4 atom
    # indices but the central atom may be in any column.  Match each itp line
    # against the canonical (center, frozenset(others)) tuples derived from
    # MoleculeGraph: we accept the line if exactly one canonical entry has
    # the same set of 4 atoms with one of them as center.
    canonical_to_idx = {(imp[0], frozenset(imp[1:])): i for i, imp in enumerate(impropers_canon)}
    impropertorsion_k = np.zeros((len(impropers_canon), 1), dtype=np.float64)
    for idxs, terms in improper_terms.items():
        total_k = sum(k for _, k in terms)
        idx_set = set(idxs)
        for atom in idxs:
            key = (atom, frozenset(idx_set - {atom}))
            if key in canonical_to_idx:
                impropertorsion_k[canonical_to_idx[key], 0] += total_k
                break

    return {
        "sigma": sigma,
        "epsilon": epsilon,
        "bond_k": bond_k,
        "bond_length": bond_length,
        "angle_k": angle_k,
        "angle_theta": angle_theta,
        "propertorsion_k": propertorsion_k,
        "impropertorsion_k": impropertorsion_k,
    }


# ---------------------------------------------------------------------------
# Per-mol worker
# ---------------------------------------------------------------------------
def process_one(uuid: str, mapped_smiles: str, coords: np.ndarray, work_root: str):
    """Build mol from mapped_smiles + coords, run AM1BCC and GAFF2, return arrays dict.

    coords shape: (n_atoms, 3) in Å.
    """
    mol = Molecule.from_mapped_smiles(mapped_smiles)
    conf = Conformer(coords=coords.astype(np.float64), symbols=mol.atomic_symbols)
    mol.append_conformers(conf)
    safe_name = uuid.replace("/", "_")
    mol.name = safe_name

    work_dir = os.path.join(work_root, safe_name)
    os.makedirs(work_dir, exist_ok=True)
    try:
        partial_charges = assign_am1bcc_charges(
            mol,
            work_dir=os.path.join(work_dir, "charges"),
            conf_id=0,
        )
        tfs = run_gaff2_acpype(mol, os.path.join(work_dir, "ff"), safe_name, conf_id=0)

        arrays = gaff2_params_to_arrays(tfs, mol)
        arrays["atomic_charge"] = np.asarray(partial_charges, dtype=np.float64).reshape(-1, 1)
        arrays["coords"] = coords.astype(np.float64).reshape(1, -1, 3)
        return arrays
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-csv", required=True)
    p.add_argument("--src-h5", required=True)
    p.add_argument("--out-h5", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    df = pd.read_csv(args.src_csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    logger.info(f"processing {len(df)} molecules")

    work_root = tempfile.mkdtemp(prefix="gaff2_h5_")
    succeeded_uuids = []
    failed = []
    try:
        with h5py.File(args.src_h5, "r") as src, h5py.File(args.out_h5, "w") as out_h5:
            for i, row in enumerate(df.itertuples(index=False)):
                uuid = row.uuid
                raw_coords = np.asarray(src[uuid]["coords"][()], dtype=np.float64)
                if raw_coords.ndim == 3:
                    coords = raw_coords[0]
                else:
                    coords = raw_coords
                try:
                    arrays = process_one(uuid, row.mapped_isomeric_smiles, coords, work_root)
                    grp = out_h5.create_group(uuid)
                    for k, v in arrays.items():
                        grp.create_dataset(k, data=v)
                    # copy hessian from source h5
                    if "hessian" in src[uuid]:
                        grp.create_dataset("hessian", data=src[uuid]["hessian"][()])
                    succeeded_uuids.append(uuid)
                except Exception:  # pylint: disable=broad-except
                    logger.warning(f"FAILED {uuid}: {traceback.format_exc().splitlines()[-1]}")
                    failed.append(uuid)
                if (i + 1) % 50 == 0:
                    logger.info(f"  done {i + 1}/{len(df)} (failed={len(failed)})")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    out_df = df[df["uuid"].isin(set(succeeded_uuids))].copy()
    out_df["h5_file"] = os.path.basename(args.out_h5)
    out_df.to_csv(args.out_csv, index=False)
    logger.info(f"wrote {len(out_df)} rows -> {args.out_csv}")
    logger.info(f"wrote {len(succeeded_uuids)} groups -> {args.out_h5}")
    logger.info(f"failed: {len(failed)}")


if __name__ == "__main__":
    main()
