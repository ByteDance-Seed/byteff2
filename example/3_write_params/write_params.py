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

import ase.io as aio
import numpy as np

from byteff2.train.utils import get_nb_params, load_model
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path


def write_gro(mol: Molecule, save_path: str):
    atoms_gro = mol.conformers[0].to_ase_atoms()
    atoms_gro.set_array('residuenames', np.array([mol.name] * mol.natoms))
    aio.write(save_path, atoms_gro)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol_name', type=str, default='AFGBL')
    parser.add_argument('--mapped_smiles',
                        type=str,
                        default='[O:1]=[C:2]1[O:3][C:4]([H:8])([H:9])[C:5]([H:10])([H:11])[C@@:6]1([F:7])[H:12]')
    parser.add_argument('--out_dir', type=str, default='./params_results')
    args = parser.parse_args()
    out_dir = args.out_dir

    # load model
    model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
    model = load_model(os.path.dirname(model_dir))

    mol = Molecule.from_mapped_smiles(args.mapped_smiles, nconfs=1)
    mol.name = args.mol_name
    metadata, params, tfs, mol = get_nb_params(model, mol)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(f'{out_dir}/{mol.name}.json'):
        os.remove(f'{out_dir}/{mol.name}.json')
    tfs.write_itp(f'{out_dir}/{mol.name}.itp', separated_atp=True)
    write_gro(mol, f'{out_dir}/{mol.name}.gro')
    with open(f'{out_dir}/{mol.name}.json', 'w') as f:
        json.dump(params, f, indent=2)
    with open(f'{out_dir}/{mol.name}_nb_params.json', 'w') as file:
        nb_params = {'metadata': metadata}
        json.dump(nb_params, file, indent=2)


if __name__ == '__main__':
    main()
