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

import torch
from tqdm import tqdm

from byteff2.data import GraphData, collate_data
from byteff2.train import load_model
from byteff2.train.utils import load_model
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path


def calc_similarity(test_mol: Molecule):

    # load model
    model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
    model = load_model(os.path.dirname(model_dir))

    cache_file = 'atom_emb_trained.pt'
    if not os.path.exists(cache_file):

        # load mols in the training set
        with open('trained_mols.json') as f:
            trained_mols = json.load(f)
        atom_embeddings = {1: [], 3: [], 6: [], 7: [], 8: [], 9: [], 15: [], 16: [], 17: []}

        # compute atom embeddings for each atom type
        for mps in tqdm(trained_mols.values(), desc='processing training set mols'):
            data = GraphData(name='test', mapped_smiles=mps)
            data = collate_data([data])
            node_h = model.graph_block(data)[0].clone().detach()
            node_h /= torch.linalg.norm(node_h, dim=-1, keepdim=True)
            mol = Molecule.from_mapped_smiles(mps)
            for atn, atemb in zip(mol.atomic_numbers, node_h):
                atom_embeddings[atn].append(atemb.clone().detach().unsqueeze(0))
        for k, v in atom_embeddings.items():
            atom_embeddings[k] = torch.concat(v, dim=0)

        torch.save(atom_embeddings, cache_file)

    else:
        atom_embeddings = torch.load(cache_file)

    data = GraphData(name='test', mapped_smiles=test_mol.get_mapped_smiles())
    data = collate_data([data])
    node_h = model.graph_block(data)[0]
    node_h /= torch.linalg.norm(node_h, dim=-1, keepdim=True)

    sims = []
    for atn, atemb in zip(test_mol.atomic_numbers, node_h):
        cos_sim = torch.sum(atom_embeddings[atn] * atemb.unsqueeze(0), dim=-1)
        cos_sim = torch.mean(torch.topk(cos_sim, k=5, dim=0)[0])
        sims.append(cos_sim.item())
    sim = min(sims)
    return sim


if __name__ == '__main__':

    # construct a molecule to compute similarity with molecules in the training set
    smiles = 'ClC(C(Cl)(Cl)Cl)C(Cl)(Cl)Cl'
    test_mol = Molecule.from_smiles(smiles)

    ## Molecule can also be constructed from sdf file, e.g.
    # test_mol = Molecule.from_sdf('{name}.sdf')

    sim = calc_similarity(test_mol)
    print(sim)
