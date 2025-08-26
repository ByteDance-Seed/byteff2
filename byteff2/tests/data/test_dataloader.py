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
import random

import torch
from torch.utils.data import DataLoader

from byteff2.data import collate_data
from byteff2.data.data import _count_idx
from byteff2.utils.definitions import MMTERM_WIDTH, MMTerm
from bytemol.utils import get_data_file_path

from .test_dataset import create_mono_dataset

logger = logging.getLogger(__name__)

meta_path = get_data_file_path("monomer_100/meta.txt", "byteff2.tests.testdata")
json_fp = get_data_file_path("monomer_100/monomer_100.json", "byteff2.tests.testdata")
hdf5_fp = get_data_file_path("monomer_100/monomer_100.h5", "byteff2.tests.testdata")


def test_dataloader_mono(tmp_path):

    _, dataset = create_mono_dataset(tmp_path)

    bs = 40
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collate_data)
    batched_data = next(iter(dataloader))

    # test collate
    nums = torch.sum(batched_data.counts, dim=0)

    assert batched_data.node_features.shape == (nums[_count_idx['node']], 5)
    assert batched_data.edge_features.shape == (nums[_count_idx['edge']], 2)

    for term in MMTerm:
        assert batched_data[f'inc_node_{term.name}'].shape[0] == batched_data[f'inc_edge_{term.name}'].shape[0] == nums[
            _count_idx[term.name]]
        assert batched_data[f'inc_node_{term.name}'].shape[
            1] == batched_data[f'inc_edge_{term.name}'].shape[1] + 1 == MMTERM_WIDTH[term]

    assert batched_data.inc_node_nonbonded14.shape == (nums[_count_idx['nonbonded14']], 2)
    assert batched_data.inc_node_nonbonded_all.shape == (nums[_count_idx['nonbonded_all']], 2)

    idx = random.randint(1, bs - 1)
    tdata = dataset[idx]
    assert (batched_data.counts[idx] == tdata.counts[0]).all()
    node_inc = torch.cumsum(batched_data.counts[:, _count_idx['node']], 0)[idx - 1]
    edge_inc = torch.cumsum(batched_data.counts[:, _count_idx['edge']], 0)[idx - 1]
    for term in MMTerm:
        nums = torch.cumsum(batched_data.counts[:, _count_idx[term.name]], 0)
        assert (batched_data[f'inc_node_{term.name}'][nums[idx - 1]:nums[idx]] -
                node_inc == tdata[f'inc_node_{term.name}']).all()
        assert (batched_data[f'inc_edge_{term.name}'][nums[idx - 1]:nums[idx]] -
                edge_inc == tdata[f'inc_edge_{term.name}']).all()

    assert (batched_data.counts[idx] == tdata.counts[0]).all()
    node_inc = torch.cumsum(batched_data.counts[:, _count_idx['node']], 0)
    assert (batched_data.coords[node_inc[idx - 1]:node_inc[idx]] == tdata.coords).all()
    assert (batched_data.forces[node_inc[idx - 1]:node_inc[idx]] == tdata.forces).all()
    assert (batched_data.energy[idx] == tdata.energy[0]).all()
