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

from byteff2.data import GraphData
from byteff2.model.ff_layers import PreChargeVolume
from byteff2.model.graph_block import Graph2DBlock


@pytest.mark.parametrize('feature_config', [{
    'atom_embedding_dim': 16,
    'connectivity_embedding_dim': 8,
    'ring_con_embedding_dim': 8,
    'min_ring_size_embedding_dim': 8,
    'fm_chg_embedding_dim': 8,
    'bond_ring_embedding_dim': 8,
    'bond_order_embedding_dim': 8,
    'scale_grad_by_freq': False,
    'node_mlp_dims': [32, 32, 3],
    'edge_mlp_dims': [32, 32, 3],
    'act': 'gelu'
}])
@pytest.mark.parametrize('gnn_config', [{
    'gnn_type': 'GINE',
    'gnn_dims': [32, 32, 3],
    'jk': None,
    'act': 'gelu',
}])
@pytest.mark.parametrize('mapped_smiles',
                         ['[O:1]([H:2])[H:3]', '[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]'])
def test_symmetry(feature_config, gnn_config, mapped_smiles):

    data = GraphData('test', mapped_smiles)
    model = Graph2DBlock(feature_config, gnn_config)

    x_h, e_h, _ = model(data)

    premm_layer = PreChargeVolume(x_h.shape[1], e_h.shape[1])

    params = premm_layer(data, x_h, e_h)

    equi_node = data.inc_node_equiv.tolist()
    equi_edge = data.inc_edge_equiv.tolist()

    node_feature, edge_feature = model.feature_layer(data)

    node_dict, edge_dict = {}, {}
    for i, equiv in enumerate(equi_node):
        if equiv not in node_dict:
            node_dict[equiv] = node_feature[i]
        else:
            assert torch.allclose(node_feature[i], node_dict[equiv])
    for i, equiv in enumerate(equi_edge):
        if equiv not in edge_dict:
            edge_dict[equiv] = edge_feature[i]
        else:
            assert torch.allclose(edge_feature[i], edge_dict[equiv])

    x_h, e_h, _ = model.gnn_layer(data, node_feature, edge_feature)
    node_dict, edge_dict = {}, {}
    for i, equiv in enumerate(equi_node):
        if equiv not in node_dict:
            node_dict[equiv] = x_h[i]
        else:
            assert torch.allclose(x_h[i], node_dict[equiv], atol=1e-6)
    for i, equiv in enumerate(equi_edge):
        if equiv not in edge_dict:
            edge_dict[equiv] = e_h[i]
        else:
            assert torch.allclose(e_h[i], edge_dict[equiv], atol=1e-6)

    charge_dict = {}
    for i, equiv in enumerate(equi_node):
        if equiv not in charge_dict:
            charge_dict[equiv] = params['PreChargeVolume.charges'][i]
        else:
            assert torch.allclose(params['PreChargeVolume.charges'][i], charge_dict[equiv], atol=1e-6)

    # test total charge

    formal_charges = data.node_features[:, 2]
    assert (params['PreChargeVolume.charges'].sum() - sum(formal_charges)).abs() < 1e-6
