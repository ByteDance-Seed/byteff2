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

import logging
from typing import Dict, Iterable, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor

logger = logging.getLogger(__name__)
rdDepictor.SetPreferCoordGen(True)

##########################################
##           read-only plot
##########################################

DEFAULT_PLOT_SHAPE = (600, 600)


def _remove_implicit_hs(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict]:
    _mol = Chem.Mol(mol)
    for atom in _mol.GetAtoms():
        atom.SetIntProp("OldIndex", atom.GetIdx())
    _mol = Chem.RemoveHs(_mol)
    id_map = dict()
    for atom in _mol.GetAtoms():
        id_map[atom.GetIntProp("OldIndex")] = atom.GetIdx()
    return _mol, id_map


def show_mol(mol: Chem.Mol,
             size: Tuple[int, int] = DEFAULT_PLOT_SHAPE,
             *,
             highlight: Iterable = None,
             remove_h: bool = False,
             plot_kekulize: bool = False,
             idx_base_1: bool = False):
    '''draw molecule (copy) with highlight'''

    _mol = Chem.Mol(mol)

    _highlight = []
    base = 1 if idx_base_1 else 0
    for atom in _mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx() + base))

    if remove_h:
        _mol, id_map = _remove_implicit_hs(_mol)
        if highlight is not None:
            for k in highlight:
                if k in id_map:
                    _highlight.append(id_map[k])
    else:
        if highlight is not None:
            _highlight = list(highlight)

    AllChem.Compute2DCoords(_mol)
    highlight_bond = []
    if _highlight:
        for i in range(len(_highlight) - 1):
            for j in range(i + 1, len(_highlight)):
                bond = _mol.GetBondBetweenAtoms(_highlight[i], _highlight[j])
                if bond is not None:
                    highlight_bond.append(bond.GetIdx())

    img = Draw.MolToImage(_mol,
                          size=size,
                          kekulize=plot_kekulize,
                          fitImage=True,
                          highlightAtoms=_highlight,
                          highlightBonds=highlight_bond)

    return img
