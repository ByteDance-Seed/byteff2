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

"""Smoke tests covering bytemol conformer generation for molecules with
hypervalent atoms (e.g. sulfoxide) where MMFF parameters can be missing
and the new xtb / fallback embedding path must be exercised.
"""

import os

import numpy as np

from byteff2.bytemol.core import Molecule
from byteff2.bytemol.core.rkutil.conformer import find_hypervalent_centers


SULFOXIDE_MAPPED_SMILES = (
    "[C:1]([c:2]1[c:3]([H:18])[c:4]([C:11]([C:12]([H:24])([H:25])[H:26])"
    "([C:13]([H:27])([H:28])[H:29])[C:14]([H:30])([H:31])[H:32])"
    "[c:5]([S@@+:8]([O-:9])[C:10]([H:21])([H:22])[H:23])[c:6]([H:19])"
    "[c:7]1[H:20])([H:15])([H:16])[H:17]"
)

PF6_MAPPED_SMILES = "[F:1][P-:2]([F:3])([F:4])([F:5])([F:6])[F:7]"


def test_from_smiles_with_sulfoxide(tmp_path):
    mol = Molecule.from_mapped_smiles(SULFOXIDE_MAPPED_SMILES, name="debug", nconfs=1)
    assert mol.natoms == 32
    assert mol.nconfs == 1
    assert mol.conformers[0].coords.shape == (32, 3)

    out = tmp_path / "debug.xyz"
    mol.to_xyz(str(out))
    assert os.path.exists(out)
    assert out.stat().st_size > 0


def test_from_smiles_with_pf6(tmp_path):
    """PF6- (P degree=6) exercises the hypervalent fallback paths in
    ``generate_confs`` (RDKit DG fails -> ``_embed_hypervalent_conformer``)
    and in ``opt_confs`` (no MMFF params + hypervalent -> skip FF opt)."""

    mol = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, name="PF6", nconfs=1)
    assert mol.natoms == 7
    assert mol.nconfs == 1

    coords = mol.conformers[0].coords
    assert coords.shape == (7, 3)
    # Embedding actually produced a non-degenerate geometry.
    assert not np.allclose(coords, 0.0)

    # Hypervalent detection should report the P center with degree 6.
    centers = find_hypervalent_centers(mol.rkmol)
    assert len(centers) == 1
    info = centers[0]
    assert info["symbol"] == "P"
    assert info["degree"] == 6
    assert info["charge"] == -1
    assert len(info["neighbors"]) == 6

    out = tmp_path / "pf6.xyz"
    mol.to_xyz(str(out))
    assert os.path.exists(out)
    assert out.stat().st_size > 0
