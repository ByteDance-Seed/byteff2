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

"""Tests for ``GraphData`` ``patch_bond_mask`` / ``patch_angle_mask`` produced
on hypervalent molecules (P/S with degree >= 5), which gate the
``MMBonded.patch_bonded_terms`` substitution path.
"""

import torch

from byteff2.bytemol.core import Molecule
from byteff2.data.data import GraphData


def _hypervalent_atom_indices(mol, atomic_nums=(15, 16), min_degree=5):
    return {
        atom.GetIdx()
        for atom in mol.rkmol.GetAtoms()
        if atom.GetDegree() >= min_degree and atom.GetAtomicNum() in atomic_nums
    }


def test_patch_masks_on_hypervalent_sulfur():
    """A hypervalent S center should mark all incident bonds and centered
    angles in the patch masks.
    """
    smiles = "[S]1(F)(F)(F)(F)OCCO1"
    mol = Molecule.from_smiles(smiles, nconfs=1, name="hyperS")
    data = GraphData(mol.name, mol.get_mapped_smiles(isomeric=False), record_nonbonded_all=False)

    assert hasattr(data, "patch_bond_mask")
    assert hasattr(data, "patch_angle_mask")

    bond_mask = data.patch_bond_mask
    angle_mask = data.patch_angle_mask
    assert bond_mask.dtype == torch.float32
    assert angle_mask.dtype == torch.float32
    assert bond_mask.shape[0] == data.inc_node_bond.shape[0]
    assert angle_mask.shape[0] == data.inc_node_angle.shape[0]

    special = _hypervalent_atom_indices(mol)
    assert special, "expected at least one hypervalent atom in the test SMILES"

    # bond mask: 1 iff one endpoint is hypervalent
    expected_bond = torch.tensor(
        [float(bond[0].item() in special or bond[1].item() in special) for bond in data.inc_node_bond],
        dtype=torch.float32,
    )
    assert torch.equal(bond_mask, expected_bond)
    assert expected_bond.sum() > 0

    # angle mask: 1 iff the central atom is hypervalent
    expected_angle = torch.tensor(
        [float(angle[1].item() in special) for angle in data.inc_node_angle],
        dtype=torch.float32,
    )
    assert torch.equal(angle_mask, expected_angle)
    assert expected_angle.sum() > 0


def test_patch_masks_on_hypervalent_pf6():
    """The PF6- anion has a hexacoordinate phosphorus center: every P-F bond
    and every F-P-F angle should be marked in the patch masks.
    """
    smiles = "F[P-](F)(F)(F)(F)F"
    mol = Molecule.from_smiles(smiles, nconfs=1, name="pf6")
    data = GraphData(mol.name, mol.get_mapped_smiles(isomeric=False), record_nonbonded_all=False)

    bond_mask = data.patch_bond_mask
    angle_mask = data.patch_angle_mask
    assert bond_mask.shape[0] == data.inc_node_bond.shape[0]
    assert angle_mask.shape[0] == data.inc_node_angle.shape[0]

    special = _hypervalent_atom_indices(mol, atomic_nums=(15,))
    assert len(special) == 1, "PF6- should have a single hypervalent P center"

    # PF6-: 6 P-F bonds, all marked
    assert int(bond_mask.sum().item()) == data.inc_node_bond.shape[0] == 6

    # 15 unordered F-P-F angles (C(6, 2)), all centered on the P atom
    assert int(angle_mask.sum().item()) == data.inc_node_angle.shape[0] == 15


def test_patch_masks_zero_on_normal_molecule():
    """Molecules without hypervalent P/S should produce all-zero patch masks."""
    smiles = "CCO"
    mol = Molecule.from_smiles(smiles, nconfs=1, name="ethanol")
    data = GraphData(mol.name, mol.get_mapped_smiles(isomeric=False), record_nonbonded_all=False)

    assert torch.all(data.patch_bond_mask == 0)
    assert torch.all(data.patch_angle_mask == 0)
