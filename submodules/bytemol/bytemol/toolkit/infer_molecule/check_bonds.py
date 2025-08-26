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

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.topology.guessers import guess_bonds

from bytemol.core import Molecule
from bytemol.utils import get_data_file_path

bl_data = pd.read_csv(get_data_file_path('bond_length_ref.csv', 'bytemol.toolkit.infer_molecule'))
bond_length_ref = {}
for i, j, l in zip(bl_data['elem_i'], bl_data['elem_j'], bl_data['length']):
    bond_length_ref[(i, j)] = l


def check_broken_bonds(mol: Molecule, conf_id=0, rtol=0.2) -> set[tuple[int]]:
    bonds = mol.get_bonds()
    coords = mol.conformers[conf_id].coords
    dist = np.linalg.norm(coords[[b[0] for b in bonds]] - coords[[b[1] for b in bonds]], axis=-1)
    broken_bonds = set()
    for b, d in zip(bonds, dist):
        ref_length = bond_length_ref[tuple(sorted([mol.atomic_numbers[b[0]], mol.atomic_numbers[b[1]]]))]
        if (d - ref_length) / ref_length > rtol:
            broken_bonds.add(b)
    return broken_bonds


def check_new_bonds(mol: Molecule, conf_id=0, vdwradii=None, fudge_factor=0.55, lower_bound=0.1) -> set[tuple[int]]:
    vdwradii = {'S': 2.4, 'BR': 1.94, 'I': 2.2} if vdwradii is None else vdwradii

    rkmol = mol.to_rkmol(conf_id=conf_id)
    for atom in rkmol.GetAtoms():
        atom.SetProp('_TriposAtomType', atom.GetSymbol().upper())
    u = mda.Universe(rkmol)
    bonds = guess_bonds(u.atoms,
                        u.atoms.positions,
                        vdwradii=vdwradii,
                        fudge_factor=fudge_factor,
                        lower_bound=lower_bound)
    return set(bonds) - set(mol.get_bonds())
