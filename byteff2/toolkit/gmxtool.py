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
import torch

from byteff2.data import GraphData
from byteff2.model.ff_layers import PreMMBondedConj
from byteff2.model.ff_layers.mm_bonded import PROPERTORSION_TERMS
from bytemol.core import Molecule
from bytemol.toolkit.gmxtool.topparse import (AngleTypeEnum, BondTypeEnum, DihedralTypeEnum, LJCombinationRuleEnum,
                                              NonbondedFunctionEnum, PairTypeEnum, RecordAngle, RecordAtom,
                                              RecordAtomType, RecordBond, RecordDihedral, RecordMoleculeType,
                                              RecordPair, Records, RecordSection, RecordText, TopoDefaults,
                                              TopoFullSystem)
from bytemol.units import simple_unit as unit


def ffparams_to_tfs(ffparams: dict[str, torch.Tensor], data: GraphData, mol: Molecule, mol_name='MOL'):

    if 'PreMMBondedConj.bond_k1' in ffparams:
        use_conj = True
    else:
        assert 'MMBondedConj.bond_k' in ffparams
        use_conj = False

    records = Records()
    comment = f"; ITP file created by ByteFF-ML, {datetime.now()}"
    record = RecordText(text="", comment=comment)
    records.all.append(record)

    # atomtypes, atoms
    atomtypes = []
    atoms = []
    indices = list(range(mol.natoms))
    sigma_list = ffparams['PreLJEs.sigma'].flatten().tolist()
    epsilon_list = ffparams['PreLJEs.epsilon'].flatten().tolist()
    charge_list = ffparams['PreLJEs.charge'].flatten().tolist()
    for i, (atomidx, sigma, epsilon, charge) in enumerate(zip(indices, sigma_list, epsilon_list, charge_list)):
        atom = mol.rkmol.GetAtomWithIdx(atomidx)
        element = atom.GetSymbol()
        at_num = atom.GetAtomicNum()
        name = f"{element.lower()}{atomidx}bf"

        atom_type = RecordAtomType(name=name, at_num=at_num, V=unit.A_to_nm(sigma), W=unit.kcal_to_kJ(epsilon))
        atomtypes.append(atom_type)

        mass = atom.GetMass()
        atom = RecordAtom(nr=atomidx + 1,
                          atype=name,
                          resnr=1,
                          residue="UNL",
                          atom=name[:-2],
                          cgnr=atomidx + 1,
                          charge=charge,
                          mass=mass)
        atoms.append(atom)

    records.all.append(RecordSection(section="atomtypes"))
    records.all += atomtypes
    records.all.append(RecordSection(section="moleculetype"))
    records.all.append(RecordMoleculeType(name=mol_name, nrexcl=3))
    records.all.append(RecordSection(section="atoms"))
    records.all += atoms

    # bonds
    indices = data.inc_node_bond.tolist()
    if use_conj:
        k1, k2 = ffparams['PreMMBondedConj.bond_k1'], ffparams['PreMMBondedConj.bond_k2']
        b1, b2 = PreMMBondedConj.bond_b1, PreMMBondedConj.bond_b2
        bond_k_list = (k1 + k2).flatten().tolist()
        bond_l_list = ((k1 * b1 + k2 * b2) / (k1 + k2)).flatten().tolist()
    else:
        bond_k_list = ffparams['PreMMBonded.bond_k'].flatten().tolist()
        bond_l_list = ffparams['PreMMBonded.bond_r0'].flatten().tolist()
    bonds = []
    for i, atomidx in enumerate(indices):
        bond = RecordBond(ai=atomidx[0] + 1,
                          aj=atomidx[1] + 1,
                          funct=BondTypeEnum.BOND,
                          c0=unit.A_to_nm(bond_l_list[i]),
                          c1=unit.kcal_mol_A2_to_kJ_mol_nm2(bond_k_list[i]))
        bonds.append(bond)
    if bonds:
        records.all.append(RecordSection(section="bonds"))
        records.all += bonds

    # angles
    indices = data.inc_node_angle.tolist()
    if use_conj:
        k1, k2 = ffparams['PreMMBondedConj.angle_k1'], ffparams['PreMMBondedConj.angle_k2']
        b1, b2 = PreMMBondedConj.angle_b1, PreMMBondedConj.angle_b2
        angle_k_list = (k1 + k2).flatten().tolist()
        angle_t_list = torch.clamp(torch.rad2deg((k1 * b1 + k2 * b2) / (k1 + k2)), max=180. - 1e-4).flatten().tolist()
    else:
        angle_k_list = ffparams['PreMMBonded.angle_k'].flatten().tolist()
        angle_t_list = ffparams['PreMMBonded.angle_d0'].flatten().tolist()
    angles = []
    for i, atomidx in enumerate(indices):
        angle = RecordAngle(ai=atomidx[0] + 1,
                            aj=atomidx[1] + 1,
                            ak=atomidx[2] + 1,
                            funct=AngleTypeEnum.ANGLE,
                            c0=angle_t_list[i],
                            c1=unit.kcal_to_kJ(angle_k_list[i]))
        angles.append(angle)
    if angles:
        records.all.append(RecordSection(section="angles"))
        records.all += angles

    # propers
    indices = data.inc_node_proper.tolist()
    if use_conj:
        proper_k_list = ffparams['PreMMBondedConj.proper_k'].tolist()
    else:
        proper_k_list = ffparams['MMBondedConj.proper_k'].tolist()
    propers = []
    for i, atomidx in enumerate(indices):
        for ip, period in enumerate(range(PROPERTORSION_TERMS)):
            record = RecordDihedral(ai=atomidx[0] + 1,
                                    aj=atomidx[1] + 1,
                                    ak=atomidx[2] + 1,
                                    al=atomidx[3] + 1,
                                    funct=DihedralTypeEnum.MULTIPLE_PROPER,
                                    c0=(period % 2) * 180.,
                                    c1=unit.kcal_to_kJ(proper_k_list[i][ip]),
                                    c2=period + 1)
            propers.append(record)
    if propers:
        records.all.append(RecordSection(section="dihedrals"))
        records.all += propers

    # impropers
    indices = data.inc_node_improper.tolist()
    if use_conj:
        improper_k_list = ffparams['PreMMBondedConj.improper_k'].flatten().tolist()
    else:
        improper_k_list = ffparams['MMBondedConj.improper_k'].flatten().tolist()
    impropers = []
    seqs = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for i, atomidx in enumerate(indices):
        for s in seqs:
            ijkl = (atomidx[0], atomidx[1 + s[0]], atomidx[1 + s[1]], atomidx[1 + s[2]])
            if improper_k_list[i] > 1e-4:
                record = RecordDihedral(ai=ijkl[0] + 1,
                                        aj=ijkl[1] + 1,
                                        ak=ijkl[2] + 1,
                                        al=ijkl[3] + 1,
                                        funct=DihedralTypeEnum.PERIODIC_IMPROPER,
                                        c0=180.,
                                        c1=unit.kcal_to_kJ(improper_k_list[i]) / 3,
                                        c2=2)
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
            f"gmx insert-molecules -f {self.input_gro} -ci {gro} -o {self.output_gro} -nmol {num} -try {try_count}")
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
        with open(file, 'w') as f:
            f.write(self.export)

    @property
    def input_gro(self):
        return f"conf_{self.index}.gro"

    @property
    def output_gro(self):
        return f"conf_{self.index+1}.gro"

    @property
    def export(self):
        return '\n'.join(self.script)