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

import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from byteff2.bytemol.core import Molecule
from byteff2.data import GraphData
from byteff2.model.ff_layers.mm_bonded import PreMMBondedConj
from byteff2.model.ff_layers.utils import get_angle_vec, get_distance_vec


class XTBCalculator:
    """xtb GFN2 geometry optimizer.

    Runs ``xtb --gfn 2 --opt`` on the input molecule and returns a copy with
    ``conformers[conf_idx].coords`` replaced by the optimized geometry.
    """

    def __init__(self, xtb_binary: str = "xtb"):
        self.xtb_binary = xtb_binary

    @staticmethod
    def _resolve_binary(xtb_binary: str) -> str:
        path = shutil.which(xtb_binary)
        if path is None:
            raise RuntimeError(
                f"xtb binary {xtb_binary!r} not found on PATH; install xtb (>=6.7) or pass xtb_binary=..."
            )
        return path

    @staticmethod
    def _run(cmd, cwd):
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"xtb command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        return proc

    @staticmethod
    def _read_xyz(path: str) -> np.ndarray:
        with open(path) as f:
            nat = int(f.readline().split()[0])
            f.readline()
            coords = np.zeros((nat, 3), dtype=np.float64)
            for i in range(nat):
                p = f.readline().split()
                coords[i] = [float(p[1]), float(p[2]), float(p[3])]
        return coords

    def run_opt(self, mol, conf_idx: int = 0):
        """Optimize geometry via ``xtb --gfn 2 --opt``.

        Args:
            mol: bytemol Molecule object (must have at least one conformer).
            conf_idx: Conformer index to optimize.

        Returns:
            A copy of ``mol`` with conformer ``conf_idx`` coords replaced by
            the GFN2-optimized geometry.

        Raises:
            RuntimeError: If the molecular topology changed during optimization
                (i.e. bonds broken or new bonds formed).
        """
        from byteff2.bytemol.toolkit.infer_molecule import check_broken_bonds, check_new_bonds

        bin_path = self._resolve_binary(self.xtb_binary)
        charge = int(round(sum(mol.formal_charges)))
        uhf = 0  # singlet by default; multiplets aren't a use case for hypervalent patches

        with tempfile.TemporaryDirectory(prefix="xtb_") as tmpdir:
            start_xyz = os.path.join(tmpdir, "start.xyz")
            mol.to_xyz(start_xyz, conf_id=conf_idx)
            self._run(
                [bin_path, "start.xyz", "--gfn", "2", "-c", str(charge), "-u", str(uhf), "--opt"],
                cwd=tmpdir,
            )
            coords_ang = self._read_xyz(os.path.join(tmpdir, "xtbopt.xyz"))

        mol_out = mol.copy()
        mol_out.conformers[conf_idx].coords = coords_ang

        broken = check_broken_bonds(mol_out, conf_id=conf_idx)
        new = check_new_bonds(mol_out, conf_id=conf_idx)
        if broken or new:
            raise RuntimeError(
                f"xtb gfn2 optimization changed molecular topology "
                f"(broken bonds: {sorted(broken)}, new bonds: {sorted(new)})."
            )
        return mol_out


def xtb_optimized_coords(mol: Molecule, conformer: int = 0) -> np.ndarray:
    """Return xtb-gfn2 optimized coords for ``mol``'s conformer.

    Generates a conformer first if ``mol`` has none. Used to prepare geometry
    for hypervalent-center parameter patching. Returns numpy coords of shape
    ``[n_node, 3]``.
    """
    if not mol.conformers:
        gen_mol = Molecule.from_mapped_smiles(mol.get_mapped_smiles(), nconfs=1, name=mol.name)
        mol.append_conformers(gen_mol.conformers[0])

    opt_input = mol.copy(keep_conformers=False)
    opt_input._conformers = [mol.conformers[conformer].copy()]
    opt_mol = XTBCalculator().run_opt(opt_input)
    return opt_mol.conformers[0].coords


def patch_hypervalent_bonded_params_for_write(ffparams: dict[str, torch.Tensor], data: GraphData):
    """Patch hypervalent bonded parameters for parameter-writing only.

    The model forward masks hypervalent terms with stable constants so training
    gradients are cut. For writing parameters, bond/angle equilibrium geometry
    should instead come from the xtb-optimized conformer prepared before the
    parameter-writing path. Keep this geometry-dependent write logic outside
    model code so model inference remains a pure tensor forward.

    Refuses to patch propers/impropers that touch a non-ring hypervalent atom:
    there is no sensible geometric / constant substitute for those force
    constants, so raise instead of silently zeroing them. Ring hypervalent
    torsions are still allowed and will be zeroed downstream.
    """
    bk, ak = 1000.0, 278.44
    if "coords" not in data:
        raise ValueError(
            "Hypervalent parameter patching requires optimized data.coords; "
            f"mapped_smiles={getattr(data, 'mapped_smiles', '<unknown>')}"
        )

    # Short-circuit when no bonded term needs patching: avoids the expensive
    # nonring_special check below for ordinary molecules.
    if not any(
        name in data and bool((data[name] > 0.9).any())
        for name in ("patch_bond_mask", "patch_angle_mask", "patch_proper_mask", "patch_improper_mask")
    ):
        return

    # Refuse non-ring hypervalent proper/improper torsions. ClusterData stores
    # ``mapped_smiles`` as a list of per-submolecule SMILES, with collated
    # ``inc_node_*`` indices shifted by the cumulative natoms of preceding
    # submolecules; build the global non-ring hypervalent atom set with the
    # same shift so the contract is enforced for both single-molecule and
    # cluster paths.
    mapped_smiles = getattr(data, "mapped_smiles", None)
    if isinstance(mapped_smiles, str):
        mapped_smiles_list = [mapped_smiles] if mapped_smiles else []
    elif isinstance(mapped_smiles, (list, tuple)):
        mapped_smiles_list = list(mapped_smiles)
    else:
        mapped_smiles_list = []

    nonring_special: set[int] = set()
    atom_shift = 0
    for mps in mapped_smiles_list:
        sub_mol = Molecule.from_mapped_smiles(mps)
        for atom in sub_mol.rkmol.GetAtoms():
            if atom.GetDegree() >= 5 and atom.GetAtomicNum() in (15, 16) and not atom.IsInRing():
                nonring_special.add(atom.GetIdx() + atom_shift)
        atom_shift += sub_mol.natoms

    if nonring_special:
        if "inc_node_proper" in data:
            for proper_atoms in data.inc_node_proper:
                if any(idx.item() in nonring_special for idx in proper_atoms):
                    raise ValueError(
                        "proper torsions on non-ring hypervalent atoms cannot be patched; "
                        f"mapped_smiles={mapped_smiles}"
                    )
        if "inc_node_improper" in data:
            for improper_atoms in data.inc_node_improper:
                if any(idx.item() in nonring_special for idx in improper_atoms):
                    raise ValueError(
                        f"impropers on non-ring hypervalent atoms cannot be patched; mapped_smiles={mapped_smiles}"
                    )

    coords = data.coords[:, 0, :]

    if "patch_bond_mask" in data:
        bond_indices = (data.patch_bond_mask > 0.9).nonzero(as_tuple=True)[0]
        if bond_indices.numel() > 0:
            bond_atoms = data.inc_node_bond[bond_indices].long()
            bond_lengths, _ = get_distance_vec(coords[bond_atoms[:, 0]], coords[bond_atoms[:, 1]])
            if "PreMMBonded.bond_r0" in ffparams:
                ffparams["PreMMBonded.bond_r0"][bond_indices] = bond_lengths.unsqueeze(-1)
            if "PreMMBonded.bond_k" in ffparams:
                ffparams["PreMMBonded.bond_k"][bond_indices] = bk
            if "PreMMBondedConj.bond_k1" in ffparams and "PreMMBondedConj.bond_k2" in ffparams:
                bond_k2 = (
                    (bond_lengths.unsqueeze(-1) - PreMMBondedConj.bond_b1)
                    / (PreMMBondedConj.bond_b2 - PreMMBondedConj.bond_b1)
                    * bk
                )
                ffparams["PreMMBondedConj.bond_k1"][bond_indices] = bk - bond_k2
                ffparams["PreMMBondedConj.bond_k2"][bond_indices] = bond_k2

    if "patch_angle_mask" in data:
        angle_indices = (data.patch_angle_mask > 0.9).nonzero(as_tuple=True)[0]
        if angle_indices.numel() > 0:
            angle_atoms = data.inc_node_angle[angle_indices].long()
            angles, _, _ = get_angle_vec(
                coords[angle_atoms[:, 0]],
                coords[angle_atoms[:, 1]],
                coords[angle_atoms[:, 2]],
                with_vec=False,
            )
            if "PreMMBonded.angle_d0" in ffparams:
                ffparams["PreMMBonded.angle_d0"][angle_indices] = torch.rad2deg(angles).unsqueeze(-1)
            if "PreMMBonded.angle_k" in ffparams:
                ffparams["PreMMBonded.angle_k"][angle_indices] = ak
            if "PreMMBondedConj.angle_k1" in ffparams and "PreMMBondedConj.angle_k2" in ffparams:
                angle_k2 = (
                    (angles.unsqueeze(-1) - PreMMBondedConj.angle_b1)
                    / (PreMMBondedConj.angle_b2 - PreMMBondedConj.angle_b1)
                    * ak
                )
                ffparams["PreMMBondedConj.angle_k1"][angle_indices] = ak - angle_k2
                ffparams["PreMMBondedConj.angle_k2"][angle_indices] = angle_k2

    if "patch_proper_mask" in data and "PreMMBondedConj.proper_k" in ffparams:
        ffparams["PreMMBondedConj.proper_k"] = torch.where(
            data.patch_proper_mask.unsqueeze(-1) > 0.9,
            torch.zeros_like(ffparams["PreMMBondedConj.proper_k"]),
            ffparams["PreMMBondedConj.proper_k"],
        )
    if "patch_improper_mask" in data and "PreMMBondedConj.improper_k" in ffparams:
        ffparams["PreMMBondedConj.improper_k"] = torch.where(
            data.patch_improper_mask.unsqueeze(-1) > 0.9,
            torch.zeros_like(ffparams["PreMMBondedConj.improper_k"]),
            ffparams["PreMMBondedConj.improper_k"],
        )
