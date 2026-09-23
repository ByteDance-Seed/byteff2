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

"""Tests for ``XTBCalculator`` (xtb GFN2 geometry optimizer)."""

import numpy as np

from byteff2.toolkit import XTBCalculator


def test_run_opt_pf6():
    """GFN2 optimization on PF6-, a hypervalent molecule."""
    from byteff2.bytemol.core import Molecule

    mol = Molecule.from_mapped_smiles("[F:1][P-:2]([F:3])([F:4])([F:5])([F:6])[F:7]", nconfs=1, name="PF6")
    initial = np.array(mol.conformers[0].coords, copy=True)

    opt_mol = XTBCalculator().run_opt(mol)

    # Geometry was rewritten and is non-trivial.
    assert opt_mol is not mol
    assert opt_mol.conformers[0].coords.shape == (7, 3)
    assert not np.allclose(opt_mol.conformers[0].coords, initial)

    # PF6- has 6 P-F bonds ~ 1.6 A in the optimized octahedron.
    coords = np.asarray(opt_mol.conformers[0].coords)
    p_idx = next(i for i, z in enumerate(opt_mol.atomic_numbers) if z == 15)
    pf_dists = [np.linalg.norm(coords[i] - coords[p_idx]) for i, z in enumerate(opt_mol.atomic_numbers) if z == 9]
    assert len(pf_dists) == 6
    assert all(1.45 < d < 1.75 for d in pf_dists), pf_dists


def test_run_opt_hyperS():
    """GFN2 optimization on a hypervalent sulfur cyclic molecule (S coordinated by 4 F + 2 O)."""
    from byteff2.bytemol.core import Molecule

    # Five-coordinate S in a 5-membered ring: 4 S-F + 2 S-O bonds.
    mol = Molecule.from_smiles("[S]1(F)(F)(F)(F)OCCO1", nconfs=1, name="hyperS")
    initial = np.array(mol.conformers[0].coords, copy=True)

    opt_mol = XTBCalculator().run_opt(mol)

    # Geometry was rewritten and is non-trivial.
    assert opt_mol is not mol
    assert opt_mol.conformers[0].coords.shape == initial.shape
    assert not np.allclose(opt_mol.conformers[0].coords, initial)

    coords = np.asarray(opt_mol.conformers[0].coords)
    s_idx = next(i for i, z in enumerate(opt_mol.atomic_numbers) if z == 16)

    # S-F distances: 4 bonds, ~1.6-1.8 A in hypervalent S.
    sf_dists = [np.linalg.norm(coords[i] - coords[s_idx]) for i, z in enumerate(opt_mol.atomic_numbers) if z == 9]
    assert len(sf_dists) == 4
    assert all(1.55 < d < 1.85 for d in sf_dists), sf_dists

    # S-O distances: 2 bonds (ring O), ~1.6-2.0 A.
    bonds = opt_mol.get_bonds()
    so_dists = [
        np.linalg.norm(coords[i] - coords[j])
        for i, j in bonds
        if {opt_mol.atomic_numbers[i], opt_mol.atomic_numbers[j]} == {16, 8}
    ]
    assert len(so_dists) == 2
    assert all(1.55 < d < 2.0 for d in so_dists), so_dists
