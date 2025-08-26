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

import pytest
import torch

from byteff2.data import GraphData, collate_data
from byteff2.model.graph_block import Graph2DBlock


@pytest.mark.parametrize('gnn_config', [{
    'gnn_type': 'EGT',
    'jk': 'cat',
    'heads': 4,
    'at_channels': 8,
    'gnn_dims': (32, 32, 2),
    'ffn_dims': (32, 2)
}, {
    'gnn_type': 'GINE'
}, {
    'gnn_type': 'GAT'
}])
def test_symmetry(gnn_config):

    mapped_smiles = '[O:1]([H:2])[H:3]'
    data = GraphData('test', mapped_smiles)

    model = Graph2DBlock(gnn_layer=gnn_config)

    x_h, e_h, _ = model(data)

    assert x_h.shape == (3, model.node_out_dim)
    assert e_h.shape == (2 * 2, model.edge_out_dim)
    assert torch.allclose(x_h[1], x_h[2])
    assert (torch.abs(e_h - e_h[0].view(1, -1)) < 1e-6).all()

    rep = 10
    data_list = []
    for _ in range(rep):
        data_list.append(data)

    data = collate_data(data_list)
    x_h_1, e_h_1, _ = model(data)

    for i in range(rep):
        assert torch.allclose(x_h_1[i * 3 + 1], x_h[1])
        assert torch.allclose(x_h_1[i * 3 + 2], x_h[1])
    assert (torch.abs(e_h_1 - e_h[0].view(1, -1)) < 1e-6).all()
