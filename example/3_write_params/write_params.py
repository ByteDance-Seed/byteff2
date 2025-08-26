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

import json
import os
from math import isclose

import ase.io as aio
import numpy as np

from byteff2.train.utils import get_nb_params, load_model
from byteff2.utils.mol_inventory import all_name_mapped_smiles
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path

OUTPUT_DIR = os.path.abspath("./params_results")
REF_DIR = os.path.abspath("./AFGBL")


def compare_floats(a, b, path="", *, rtol=1e-5, atol=1e-8):
    """Recursively compare floats (or nested lists) in two JSON-like objects."""
    if isinstance(a, float) and isinstance(b, float):
        if not isclose(a, b, rel_tol=rtol, abs_tol=atol):
            print(f"MISMATCH at {path}: {a} vs {b}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"LENGTH MISMATCH at {path}: {len(a)} vs {len(b)}")
            return
        for idx, (ai, bi) in enumerate(zip(a, b)):
            compare_floats(ai, bi, f"{path}[{idx}]", rtol=rtol, atol=atol)
    elif isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() | b.keys():
            if key not in a or key not in b:
                print(f"KEY MISSING at {path}: {key}")
                continue
            compare_floats(a[key], b[key], f"{path}.{key}", rtol=rtol, atol=atol)
    else:
        # Non-float scalars (int, str, bool) must be exactly equal
        if a != b:
            print(f"NON-FLOAT MISMATCH at {path}: {a} vs {b}")


def write_gro(mol: Molecule, save_path: str):
    atoms_gro = mol.conformers[0].to_ase_atoms()
    atoms_gro.set_array('residuenames', np.array([mol.name] * mol.natoms))
    aio.write(save_path, atoms_gro)


def main():
    # load model
    model_dir = get_data_file_path('optimal.pt', 'byteff2.trained_models')
    model = load_model(os.path.dirname(model_dir))
    # generate input mol
    mols = all_name_mapped_smiles
    for idx, (name, mps) in enumerate(mols.items()):
        if idx == 1:  # stop after the first two entries
            break
        mol = Molecule.from_mapped_smiles(mps, nconfs=1)
        mol.name = name
        # generate force field params
        metadata, params, tfs, mol = get_nb_params(model, mol)
        # clean old data
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if os.path.exists(f'{OUTPUT_DIR}/{mol.name}.json'):
            os.remove(f'{OUTPUT_DIR}/{mol.name}.json')
        tfs.write_itp(f'{OUTPUT_DIR}/{mol.name}.itp', separated_atp=True)
        write_gro(mol, f'{OUTPUT_DIR}/{mol.name}.gro')
        with open(f'{OUTPUT_DIR}/{mol.name}.json', 'w') as f:
            json.dump(params, f, indent=2)
        with open(f'{OUTPUT_DIR}/{mol.name}_nb_params.json', 'w') as file:
            nb_params = {'metadata': metadata}
            json.dump(nb_params, file, indent=2)
        # compare with reference json file
        with open(f'{OUTPUT_DIR}/{mol.name}.json', 'r') as f1, open(f'{REF_DIR}/{mol.name}.json') as f2:
            generated_params = json.load(f1)
            ref_params = json.load(f2)
        print(f'compare {mol.name} with reference json file')
        compare_floats(generated_params, ref_params)


if __name__ == '__main__':
    main()
