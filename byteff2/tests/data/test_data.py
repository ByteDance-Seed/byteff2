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
import torch

from byteff2.data.data import ClusterData, GraphData, MonoData
from byteff2.utils.definitions import MMTERM_WIDTH, MMTerm
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path

logger = logging.getLogger(__name__)

mono_json_fp = get_data_file_path("monomer_100/monomer_100.json", "byteff2.tests.testdata")
mono_hdf5_fp = get_data_file_path("monomer_100/monomer_100.h5", "byteff2.tests.testdata")

with open(mono_json_fp) as file:
    mono_name_smiles = json.load(file)

dimer_json_fp = get_data_file_path("dimer_100/dimer100.json", "byteff2.tests.testdata")
dimer_hdf5_fp = get_data_file_path("dimer_100/dimer100.h5", "byteff2.tests.testdata")

with open(dimer_json_fp) as file:
    dimer_name_smiles = json.load(file)


def test_graph_data():

    for name, smiles in mono_name_smiles.items():
        mol = Molecule.from_mapped_smiles(smiles, name=name)
        data = GraphData(name, smiles)

        assert data.mol_name == mol.name
        assert data.get_count('node') == mol.natoms
        assert data.get_count('edge') == len(mol.get_bonds()) * 2
        assert data.node_features.shape == (mol.natoms, 5)
        assert data.edge_features.shape == (data.get_count('edge'), 2)

        for term in MMTerm:
            assert data[f'inc_node_{term.name}'].shape[0] == data[f'inc_edge_{term.name}'].shape[0] == data.get_count(
                term.name)
            assert data[f'inc_node_{term.name}'].shape[1] == data[f'inc_edge_{term.name}'].shape[1] + 1 == MMTERM_WIDTH[
                term]

        assert data.inc_node_nonbonded14.shape == (data.get_count('nonbonded14'), 2)
        assert data.inc_node_nonbonded_all.shape == (data.get_count('nonbonded_all'), 2)


def test_momo_data():

    h5_file = h5py.File(mono_hdf5_fp, 'r')

    for max_n_confs in [5, 20]:

        for name, smiles in mono_name_smiles.items():
            mol = Molecule.from_mapped_smiles(smiles, name=name)
            dataset = h5_file[name]

            data = MonoData(name,
                            smiles,
                            confdata=dict(coords=dataset['coords'][:],
                                          energy=dataset['energy'][:],
                                          forces=dataset['forces'][:]),
                            max_n_confs=max_n_confs)

            assert data.mol_name == mol.name
            assert data.get_count('node') == mol.natoms
            assert data.get_count('edge') == len(mol.get_bonds()) * 2

            assert data.coords.shape == (mol.natoms, max_n_confs, 3)
            assert data.forces.shape == (mol.natoms, max_n_confs, 3)
            assert data.energy.shape == (1, max_n_confs)

    h5_file.close()


def test_from_dict():

    h5_file = h5py.File(mono_hdf5_fp, 'r')

    max_n_confs = 20

    names = list(mono_name_smiles)
    name = random.choice(names)
    smiles = mono_name_smiles[name]

    dataset = h5_file[name]

    data = MonoData(name,
                    smiles,
                    confdata=dict(coords=dataset['coords'][:], energy=dataset['energy'][:],
                                  forces=dataset['forces'][:]),
                    max_n_confs=max_n_confs)

    new_data = MonoData.from_dict(dict(data))
    for k, v in data.items():
        assert k in new_data
        if isinstance(v, str):
            assert v == new_data[k]
        elif isinstance(v, torch.Tensor):
            assert (v == new_data[k]).all(), k
        else:
            raise TypeError(f'Unknow type of {k}: {type(v)}')

    h5_file.close()


def test_cluster_data():

    max_n_confs = 10

    h5_file = h5py.File(dimer_hdf5_fp, 'r')
    for name, smiles in dimer_name_smiles.items():
        print(smiles)
        dataset = h5_file[name]
        data = ClusterData(name,
                           mapped_smiles=smiles,
                           max_n_confs=max_n_confs,
                           confdata=dict(
                               coords=dataset['coords'],
                               forces_cluster=dataset['forces_cluster'],
                               energy_cluster=dataset['energy_cluster'],
                           ))

        mols = [Molecule.from_mapped_smiles(s) for s in smiles]

        n_node, n_edge = 0, 0
        for i, mol in enumerate(mols):
            assert data.get_count('node', idx=i) == mol.natoms
            assert data.get_count('edge', idx=i) == len(mol.get_bonds()) * 2
            n_node += mol.natoms
            n_edge += len(mol.get_bonds()) * 2

        assert data.get_count('node', cluster=True) == n_node
        assert data.get_count('edge', cluster=True) == n_edge
        assert data.get_count('nonbonded14', cluster=True) == data.get_count('nonbonded14', idx=None).sum().item()
        assert data.get_count('nonbonded_all', cluster=True) > data.get_count('nonbonded14', idx=None).sum().item()

        assert data.coords.shape == (n_node, max_n_confs, 3)
        assert data.forces_cluster.shape == (n_node, max_n_confs, 3)
        assert data.energy_cluster.shape == (1, max_n_confs)


def test_cluster_nonbonded():

    natoms = 5
    nmols = 4
    data = ClusterData('test', mapped_smiles=['[C:1]([H:2])([H:3])([H:4])[H:5]'] * nmols)
    nonbonded_all_cluster = data.inc_node_nonbonded_all_cluster.tolist()
    for i in range(natoms * nmols):
        for j in range(natoms * nmols):
            if abs(i - j) > (natoms - 1):
                assert sorted((i, j)) in nonbonded_all_cluster
