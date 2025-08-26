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

import torch
from torch.utils.data import DataLoader

from byteff2.data import GraphData, MonoData, collate_data
from byteff2.model.ff_layers import MMBonded, PreMMBonded
from byteff2.model.graph_block import Graph2DBlock
from byteff2.tests.data.test_dataset import create_mono_dataset


def test_pre_mm_bonded():

    mapped_smiles = '[C:1](=[O:2])([C:3]([H:5])([H:6])[H:7])[H:4]'
    data = GraphData('test', mapped_smiles)

    model = Graph2DBlock(gnn_layer={'gnn_type': 'GINE'})

    x_h, e_h, _ = model(data)

    premm_layer = PreMMBonded(x_h.shape[1], e_h.shape[1])

    params = premm_layer(data, x_h, e_h)

    for p in ['bond_k', 'bond_r0', 'angle_k', 'angle_d0', 'proper_k']:
        pp = 'PreMMBonded.' + p
        assert (params[pp][-1, 0] - params[pp][-2, 0]).abs().max() < 1e-4
        assert (params[pp][-1, 0] - params[pp][-3, 0]).abs().max() < 1e-4

    assert params['PreMMBonded.improper_k'].shape == (1, 1)


def test_mm_bonded(tmp_path):

    dim = 64
    premm_layer = PreMMBonded(dim, dim)
    mm_layer = MMBonded(dim, dim)

    _, dataset = create_mono_dataset(tmp_path)
    dataloader = DataLoader(dataset, batch_size=40, shuffle=False, collate_fn=collate_data)
    data: MonoData = next(iter(dataloader))

    x_h, e_h = torch.rand((data.node_features.shape[0], dim)), torch.rand((data.edge_features.shape[0], dim))
    ff_params = premm_layer(data, x_h, e_h)
    energy, forces = mm_layer(data, x_h, e_h, ff_params)
    assert energy.shape == (data.counts.shape[0], data.coords.shape[1])
    assert forces.shape == data.coords.shape
