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

from datetime import datetime

import numpy as np
from rdkit import Chem
import torch

from byteff2.bytemol.core import Molecule
from byteff2.bytemol.toolkit.gmxtool.topparse import (
    AngleTypeEnum,
    BondTypeEnum,
    DihedralTypeEnum,
    LJCombinationRuleEnum,
    NonbondedFunctionEnum,
    PairTypeEnum,
    RecordAngle,
    RecordAtom,
    RecordAtomType,
    RecordBond,
    RecordDihedral,
    RecordMoleculeType,
    RecordPair,
    Records,
    RecordSection,
    RecordText,
    TopoDefaults,
    TopoFullSystem,
)
from byteff2.bytemol.units import simple_unit as unit
from byteff2.data import GraphData
from byteff2.model.ff_layers import PreMMBondedConj
from byteff2.model.ff_layers.mm_bonded import PROPERTORSION_TERMS


HBOND14_ANGLE_SMARTS = "[#7,#8;!H0:1]~[#15,#16;D4:2]~[#7,#8;H0:3]"
# RecordAngle.k uses GROMACS angle units: kJ mol^-1 rad^-2.
HBOND14_ANGLE_FORCE_CONSTANT_FLOOR = 600.0
_HBOND14_ANGLE_QUERY = Chem.MolFromSmarts(HBOND14_ANGLE_SMARTS)


def patch_hbond14_angle_force_constants(
    tfs: TopoFullSystem,
    mol: Molecule,
    minimum: float = HBOND14_ANGLE_FORCE_CONSTANT_FLOOR,
) -> TopoFullSystem:
    """Floor matching angle force constants in place and return ``tfs``."""
    matching_angles = {
        (min(ai, ak) + 1, aj + 1, max(ai, ak) + 1)
        for ai, aj, ak in mol.rkmol.GetSubstructMatches(_HBOND14_ANGLE_QUERY, uniquify=True)
    }

    for angle in tfs.mol_topos[0].angles:
        angle_indices = (min(angle.ai, angle.ak), angle.aj, max(angle.ai, angle.ak))
        if angle_indices in matching_angles and angle.k < minimum:
            angle.k = minimum
    return tfs


def convert_conj_to_bonded_params(ffparams: dict[str, torch.Tensor]) -> None:
    """Convert PreMMBondedConj parameters to PreMMBonded format.

    If ffparams contains PreMMBondedConj parameters, convert them:
    - bond_k1, bond_k2 -> bond_k, bond_r0
    - angle_k1, angle_k2 -> angle_k, angle_d0
    - proper_k -> proper_k
    - improper_k -> improper_k
    """

    k1 = ffparams.get("PreMMBondedConj.bond_k1")
    k2 = ffparams.get("PreMMBondedConj.bond_k2")
    if k1 is not None and k2 is not None:
        b1, b2 = PreMMBondedConj.bond_b1, PreMMBondedConj.bond_b2
        ffparams["PreMMBonded.bond_k"] = k1 + k2
        ffparams["PreMMBonded.bond_r0"] = (k1 * b1 + k2 * b2) / (k1 + k2)
        k1, k2 = ffparams["PreMMBondedConj.angle_k1"], ffparams["PreMMBondedConj.angle_k2"]
        b1, b2 = PreMMBondedConj.angle_b1, PreMMBondedConj.angle_b2
        ffparams["PreMMBonded.angle_k"] = k1 + k2
        ffparams["PreMMBonded.angle_d0"] = torch.clamp(torch.rad2deg((k1 * b1 + k2 * b2) / (k1 + k2)), max=180.0 - 1e-4)
        ffparams["PreMMBonded.proper_k"] = ffparams["PreMMBondedConj.proper_k"]
        ffparams["PreMMBonded.improper_k"] = ffparams["PreMMBondedConj.improper_k"]


_UNC_DISPLAY_CONV = {
    "PreLJEs.sigma": unit.A_to_nm,
    "PreLJEs.epsilon": unit.kcal_to_kJ,
    "PreLJEs.charges": None,
    "PreMMBonded.bond_r0": unit.A_to_nm,
    "PreMMBonded.bond_k": unit.kcal_mol_A2_to_kJ_mol_nm2,
    "PreMMBonded.angle_d0": None,
    "PreMMBonded.angle_k": unit.kcal_to_kJ,
    "PreMMBonded.proper_k": unit.kcal_to_kJ,
    "PreMMBonded.improper_k": lambda t: unit.kcal_to_kJ(t) / 3.0,
}


def _uncertainty_lists(ffparams: dict[str, torch.Tensor]):
    """Pull per-term uncertainty out of an ensemble's ffparams for itp comments.

    ``EnsembleModel`` stores, under ``ffparams["uncertainty"]``, the
    across-member relative std (``std / max(mean, 1)`` in the model's *internal*
    units) of every predicted parameter, keyed exactly like the mean tensors. The
    itp comments should be absolute std in the same display units as the written
    values, so this converts the relative std back to internal absolute std and
    then applies the same unit conversion used for the mean value. A single
    ``HybridFF`` has no such key, so this returns ``None`` and no uncertainty
    comments are emitted (single-model itps stay identical).

    Returns flat python lists per term, or ``None`` for a term the ensemble did
    not provide (e.g. a ``PreMMBondedConj`` model exposes no derived
    ``bond_k``/``bond_r0`` uncertainty), in which case that comment is skipped.
    """
    unc = ffparams.get("uncertainty")
    if not unc:
        return None

    def abs_std(key):
        rstd = unc.get(key)
        mean = ffparams.get(key)
        if rstd is None or mean is None:
            return None
        denom = torch.where(mean > 1.0, mean, torch.ones_like(mean))
        std = rstd * denom
        conv = _UNC_DISPLAY_CONV.get(key)
        return conv(std) if conv is not None else std

    def flat(key):
        t = abs_std(key)
        return t.flatten().tolist() if t is not None else None

    def rows(key):  # keep the per-periodicity axis (propers have 4 terms each)
        t = abs_std(key)
        return t.tolist() if t is not None else None

    return {
        "sigma": flat("PreLJEs.sigma"),
        "epsilon": flat("PreLJEs.epsilon"),
        "charge": flat("PreLJEs.charges"),
        "bond_r0": flat("PreMMBonded.bond_r0"),
        "bond_k": flat("PreMMBonded.bond_k"),
        "angle_d0": flat("PreMMBonded.angle_d0"),
        "angle_k": flat("PreMMBonded.angle_k"),
        "proper_k": rows("PreMMBonded.proper_k"),
        "improper_k": flat("PreMMBonded.improper_k"),
    }


def _at(lst, i):
    """Element ``i`` of a possibly-missing list (``None`` list -> ``None``)."""
    return lst[i] if lst is not None else None


def _unc_comment(pairs):
    """Format ``uncertainty a = .., b = ..`` from ``(label, value)`` pairs.

    A ``None`` value (term unavailable) is skipped; if nothing is left the whole
    comment is ``None`` so no trailing ``; uncertainty`` is written.
    """
    parts = [f"{label} = {val:.6f}" for label, val in pairs if val is not None]
    return f"uncertainty {', '.join(parts)}" if parts else None


def ffparams_to_tfs(
    ffparams: dict[str, torch.Tensor],
    data: GraphData,
    mol: Molecule,
    mol_name="MOL",
    apply_hbond14_patch: bool = True,
):

    records = Records()
    # for an ensemble this is per-parameter absolute std in itp display units
    unc = _uncertainty_lists(ffparams)
    comment = f"; ITP file created by ByteFF-ML, {datetime.now()}"
    record = RecordText(text="", comment=comment)
    records.all.append(record)

    # atomtypes, atoms
    atomtypes = []
    atoms = []
    indices = list(range(mol.natoms))
    sigma_list = ffparams["PreLJEs.sigma"].flatten().tolist()
    epsilon_list = ffparams["PreLJEs.epsilon"].flatten().tolist()
    charge_list = ffparams["PreLJEs.charges"].flatten().tolist()
    for i, (atomidx, sigma, epsilon, charge) in enumerate(zip(indices, sigma_list, epsilon_list, charge_list)):
        atom = mol.rkmol.GetAtomWithIdx(atomidx)
        element = atom.GetSymbol()
        at_num = atom.GetAtomicNum()
        name = f"{element.lower()}{atomidx}bf"

        # sigma/epsilon/charge uncertainty rides on the atomtypes line: the
        # [ atoms ] comment is overwritten by qtot in from_records, so this is
        # the only per-atom line that can carry a nonbonded uncertainty note.
        at_comment = _unc_comment([
            ("sigma", _at(unc["sigma"], i) if unc else None),
            ("epsilon", _at(unc["epsilon"], i) if unc else None),
            ("charge", _at(unc["charge"], i) if unc else None),
        ]) if unc else None
        atom_type = RecordAtomType(name=name, at_num=at_num, V=unit.A_to_nm(sigma), W=unit.kcal_to_kJ(epsilon),
                                   comment=at_comment)
        atomtypes.append(atom_type)

        mass = atom.GetMass()
        atom = RecordAtom(
            nr=atomidx + 1,
            atype=name,
            resnr=1,
            residue="UNL",
            atom=name[:-2],
            cgnr=atomidx + 1,
            charge=charge,
            mass=mass,
        )
        atoms.append(atom)

    records.all.append(RecordSection(section="atomtypes"))
    records.all += atomtypes
    records.all.append(RecordSection(section="moleculetype"))
    records.all.append(RecordMoleculeType(name=mol_name, nrexcl=3))
    records.all.append(RecordSection(section="atoms"))
    records.all += atoms

    # bonds
    indices = data.inc_node_bond.tolist()
    convert_conj_to_bonded_params(ffparams)
    bond_k_list = ffparams["PreMMBonded.bond_k"].flatten().tolist()
    bond_l_list = ffparams["PreMMBonded.bond_r0"].flatten().tolist()
    bonds = []
    for i, atomidx in enumerate(indices):
        bond = RecordBond(
            ai=atomidx[0] + 1,
            aj=atomidx[1] + 1,
            funct=BondTypeEnum.BOND,
            c0=unit.A_to_nm(bond_l_list[i]),
            c1=unit.kcal_mol_A2_to_kJ_mol_nm2(bond_k_list[i]),
            comment=_unc_comment([("r", _at(unc["bond_r0"], i)), ("k", _at(unc["bond_k"], i))]) if unc else None,
        )
        bonds.append(bond)
    if bonds:
        records.all.append(RecordSection(section="bonds"))
        records.all += bonds

    # angles
    indices = data.inc_node_angle.tolist()
    angle_k_list = ffparams["PreMMBonded.angle_k"].flatten().tolist()
    angle_t_list = ffparams["PreMMBonded.angle_d0"].flatten().tolist()
    angles = []
    for i, atomidx in enumerate(indices):
        angle = RecordAngle(
            ai=atomidx[0] + 1,
            aj=atomidx[1] + 1,
            ak=atomidx[2] + 1,
            funct=AngleTypeEnum.ANGLE,
            c0=angle_t_list[i],
            c1=unit.kcal_to_kJ(angle_k_list[i]),
            comment=_unc_comment([("theta", _at(unc["angle_d0"], i)), ("k", _at(unc["angle_k"], i))]) if unc else None,
        )
        angles.append(angle)
    if angles:
        records.all.append(RecordSection(section="angles"))
        records.all += angles

    # propers
    indices = data.inc_node_proper.tolist()
    proper_k_list = ffparams.get("PreMMBondedConj.proper_k", ffparams["PreMMBonded.proper_k"]).tolist()
    propers = []
    for i, atomidx in enumerate(indices):
        for ip, period in enumerate(range(PROPERTORSION_TERMS)):
            # proper_k uncertainty is per (proper, periodicity); match the mean's layout
            unc_k = unc["proper_k"][i][ip] if unc and unc["proper_k"] is not None else None
            record = RecordDihedral(
                ai=atomidx[0] + 1,
                aj=atomidx[1] + 1,
                ak=atomidx[2] + 1,
                al=atomidx[3] + 1,
                funct=DihedralTypeEnum.MULTIPLE_PROPER,
                c0=(period % 2) * 180.0,
                c1=unit.kcal_to_kJ(proper_k_list[i][ip]),
                c2=period + 1,
                comment=_unc_comment([("k", unc_k)]) if unc else None,
            )
            propers.append(record)
    if propers:
        records.all.append(RecordSection(section="dihedrals"))
        records.all += propers

    # impropers
    indices = data.inc_node_improper.tolist()
    improper_k_list = ffparams.get("PreMMBondedConj.improper_k", ffparams["PreMMBonded.improper_k"]).flatten().tolist()
    impropers = []
    seqs = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for i, atomidx in enumerate(indices):
        for s in seqs:
            ijkl = (atomidx[0], atomidx[1 + s[0]], atomidx[1 + s[1]], atomidx[1 + s[2]])
            if improper_k_list[i] > 1e-4:
                # the 3 permuted rows of one improper share its single uncertainty
                record = RecordDihedral(
                    ai=ijkl[0] + 1,
                    aj=ijkl[1] + 1,
                    ak=ijkl[2] + 1,
                    al=ijkl[3] + 1,
                    funct=DihedralTypeEnum.PERIODIC_IMPROPER,
                    c0=180.0,
                    c1=unit.kcal_to_kJ(improper_k_list[i]) / 3,
                    c2=2,
                    comment=_unc_comment([("k", _at(unc["improper_k"], i))]) if unc else None,
                )
                impropers.append(record)
    if impropers:
        records.all.append(RecordSection(section="dihedrals"))
        records.all += impropers

    # pairs
    pairs = set()
    pair_list = []
    for atomidx in data.inc_node_nonbonded14.tolist():
        pair = tuple(atomidx)
        if pair not in pairs:
            pairs.add(pair)
            record = RecordPair(ai=pair[0] + 1, aj=pair[1] + 1, funct=PairTypeEnum.EXTRA_LJ)
            pair_list.append(record)
    if pair_list:
        records.all.append(RecordSection(section="pairs"))
        records.all += pair_list

    tfs = TopoFullSystem.from_records(records=records.all, sort_idx=True, round_on="w")
    # amber style [ defaults ]
    td = TopoDefaults(tfs.uuid)
    assert td.nbfunc == NonbondedFunctionEnum.LENNARD_JONES
    assert td.comb_rule == LJCombinationRuleEnum.SIGMA_EPSILON
    assert td.gen_pairs == "yes"
    assert np.isclose(td.fudge_lj, 0.5)
    assert np.isclose(td.fudge_qq, 0.5), f"fudge_qq is {td.fudge_qq}"
    if apply_hbond14_patch:
        tfs = patch_hbond14_angle_force_constants(tfs, mol)
    return tfs


class GMXScript:
    def __init__(self) -> None:
        self.script = []
        self.index = 0

        add_flag = """#!/bin/bash

set -xe

# Default values
ratio=1.0

# Loop through command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ratio)
            ratio="$2"
            shift 2
            ;;
        *) # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done"""

        self.add(add_flag)
        self.used_gro = []

    def add(self, line: str):
        if not line.endswith("\n"):
            line = line + "\n"
        self.script.append(line)

    def init_gro_box(self, init_gro: str, box: float):
        self.add(f"default_box_size={box}")
        self.add('box_size=$(awk "BEGIN {print $default_box_size*$ratio}")')
        self.add(f"gmx editconf -f {init_gro} -o {self.output_gro} -box $box_size $box_size $box_size")
        self.index += 1

    def scale(self, scale: float):
        self.add(f"gmx editconf -f {self.input_gro} -o {self.output_gro} -scale {scale}")
        self.index += 1

    def insert_molecules(self, gro: str, num: int, try_count: int = 15000):
        self.add(
            f"gmx insert-molecules -f {self.input_gro} -ci {gro} -o {self.output_gro} -nmol {num} -try {try_count}"
        )
        self.used_gro.append(gro)
        self.index += 1

    def genconf(self, init_gro: str, box: int):
        assert init_gro.endswith(".gro"), "gro file is required."
        self.add(f"gmx genconf -f {init_gro} -o {self.output_gro} -nbox {box}")
        self.index += 1

    def finish(self):
        target_name = "solvent_salt.gro"
        self.add(f"mv {self.input_gro} {target_name}")
        # remove all the intermediate files
        self.add("rm -f conf_*.gro")

    def write(self, file: str):
        with open(file, "w") as f:
            f.write(self.export)

    @property
    def input_gro(self):
        return f"conf_{self.index}.gro"

    @property
    def output_gro(self):
        return f"conf_{self.index + 1}.gro"

    @property
    def export(self):
        return "\n".join(self.script)
