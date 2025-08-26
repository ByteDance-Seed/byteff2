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
import logging
import random

import h5py
from torch import Tensor

from byteff2.data import ClusterData, IMDataset, MonoData
from bytemol.utils import get_data_file_path

logger = logging.getLogger(__name__)

mono_meta_fp = get_data_file_path("monomer_100/meta.txt", "byteff2.tests.testdata")
mono_json_fp = get_data_file_path("monomer_100/monomer_100.json", "byteff2.tests.testdata")
mono_hdf5_fp = get_data_file_path("monomer_100/monomer_100.h5", "byteff2.tests.testdata")

dimer_meta_fp = get_data_file_path("dimer_100/meta.txt", "byteff2.tests.testdata")
dimer_json_fp = get_data_file_path("dimer_100/dimer100.json", "byteff2.tests.testdata")
dimer_hdf5_fp = get_data_file_path("dimer_100/dimer100.h5", "byteff2.tests.testdata")


def create_mono_dataset(path, max_n_confs=10):

    config = {
        'meta_fp': mono_meta_fp,
        'save_dir': path,
        'data_cls': 'MonoData',
        'confdata': {
            'coords': 'coords',
            'energy': 'energy',
            'forces': 'forces',
        },
        'kwargs': {
            'max_n_confs': max_n_confs
        },
    }

    # processing and saving
    IMDataset.process(config, 0)
    dataset = IMDataset(config)
    return config, dataset


def test_dataset_monodata(tmp_path):

    max_n_confs = 16
    config, dataset = create_mono_dataset(tmp_path, max_n_confs)

    idx = random.randint(1, len(dataset) - 1)
    with open(mono_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(',')

    h5 = h5py.File(mono_hdf5_fp)
    h5_dataset = h5[name]
    with open(mono_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = MonoData(name,
                     smiles,
                     confdata=dict(coords=h5_dataset['coords'][:],
                                   energy=h5_dataset['energy'][:],
                                   forces=h5_dataset['forces'][:]),
                     max_n_confs=max_n_confs)

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all()
        else:
            print(k, v)

    h5.close()

    # test loading
    dataset = IMDataset(config, processing=False)

    idx = random.randint(1, len(dataset) - 1)
    with open(mono_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(',')

    h5 = h5py.File(mono_hdf5_fp)
    h5_dataset = h5[name]
    with open(mono_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = MonoData(name,
                     smiles,
                     confdata=dict(coords=h5_dataset['coords'][:],
                                   energy=h5_dataset['energy'][:],
                                   forces=h5_dataset['forces'][:]),
                     max_n_confs=max_n_confs)

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all()
        else:
            raise TypeError(f'Unknow type of {k}: {type(v)}')

    h5.close()


def create_dimer_dataset(path, max_n_confs=10):

    config = {
        'meta_fp': dimer_meta_fp,
        'save_dir': path,
        'data_cls': 'ClusterData',
        'confdata': {
            'coords': 'coords',
            'forces_cluster': 'forces_cluster',
            'energy_cluster': 'energy_cluster',
        },
        'kwargs': {
            'max_n_confs': max_n_confs
        },
    }

    # processing and saving
    IMDataset.process(config, 0)
    dataset = IMDataset(config)
    return config, dataset


def test_dataset_dimer(tmp_path):

    max_n_confs = 50
    config, dataset = create_dimer_dataset(tmp_path, max_n_confs)

    idx = random.randint(1, len(dataset) - 1)
    with open(dimer_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(',')

    h5 = h5py.File(dimer_hdf5_fp)
    h5_dataset = h5[name]
    with open(dimer_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = ClusterData(name,
                        smiles,
                        confdata=dict(coords=h5_dataset['coords'][:],
                                      forces_cluster=h5_dataset['forces_cluster'][:],
                                      energy_cluster=h5_dataset['energy_cluster'][:]),
                        max_n_confs=max_n_confs)

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all(), k
        else:
            assert v == data[k], k

    h5.close()

    # test loading
    dataset = IMDataset(config, processing=False)

    idx = random.randint(1, len(dataset) - 1)
    with open(dimer_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(',')

    h5 = h5py.File(dimer_hdf5_fp)
    h5_dataset = h5[name]
    with open(dimer_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = ClusterData(name,
                        smiles,
                        confdata=dict(coords=h5_dataset['coords'][:],
                                      forces_cluster=h5_dataset['forces_cluster'][:],
                                      energy_cluster=h5_dataset['energy_cluster'][:]),
                        max_n_confs=max_n_confs)

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all()
        else:
            assert v == data[k], k

    h5.close()
