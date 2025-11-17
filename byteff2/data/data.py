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

import functools
import logging
from operator import itemgetter

import numpy as np
import torch
from torch import Tensor

from byteff2.utils.definitions import ELEMENT_MAP, MAX_RING_SIZE, MM_TOPO_MAP, MMTERM_WIDTH, BondOrder, MMTerm
from byteff2.utils.mol_utils import find_equivalent_index, get_ring_info, match_linear_proper
from bytemol.core import Conformer, Molecule, MoleculeGraph, rkutil
from bytemol.toolkit.infer_molecule import check_broken_bonds, check_new_bonds

logger = logging.getLogger(__name__)


@functools.cache
def _cos_linspace(j, steps):
    x = torch.linspace(0, steps / 2, steps)
    return torch.cos(x * j) + x


def fake_coords(n_node, n_conf):
    coords = torch.stack([_cos_linspace(j, n_node) for j in range(3)]).T.unsqueeze(1)  # [n_node, 1, 3]
    coords = coords.repeat(1, n_conf, 1)  # [n_node, n_conf, 3]
    return coords


_count_names = [
    'node', 'edge', 'bond', 'angle', 'proper', 'improper', 'nonbonded14', 'nonbonded_all', 'mol', 'nonbonded12',
    'nonbonded13', 'nonbonded15'
]
_count_idx = {name: idx for idx, name in enumerate(_count_names)}


class Data(dict[str, Tensor]):
    """
    Data is a dict containing features, most of which are Tensors.
    
    If one key is start by 'inc_{name}_', its value contains {name} indices and
      should be increased by the cummulation of counts[COUNT_IDX[name]] when collating.

    `coords` and `confdata` have the size of first dimension. which is nconf.
    It will be padded to `max_n_confs` and be swapped to the second dimension.
    """

    def __setattr__(self, name: str, value: Tensor):
        self[name] = value

    def __getattr__(self, name: str) -> Tensor:
        if name in self.keys():
            return self[name]
        else:
            super(Data).__getattr__(name)

    def __init__(self,
                 moldata: dict[str, np.ndarray] = None,
                 confdata: dict[str, np.ndarray] = None,
                 max_n_confs: int = 10,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32,
                 **kwargs):
        super().__init__()
        self.counts = torch.tensor([0] * len(_count_names), dtype=int_dtype).reshape(1, -1)
        self.counts_cluster = torch.tensor([0] * len(_count_names), dtype=int_dtype).reshape(1, -1)

        if moldata is not None:
            for k, v in moldata.items():
                if isinstance(v, Tensor):
                    self[k] = v
                elif isinstance(v, np.ndarray):
                    self[k] = torch.tensor(v, dtype=float_dtype)
                else:
                    raise ValueError(f'unknown kwargs {k}, {v}')

        if confdata is not None:
            coords = confdata.pop('coords')
            _coords = torch.tensor(coords, dtype=float_dtype).transpose(0, 1)  # [n_node, n_conf, 3]
            _confdata: dict[str, Tensor] = {}
            if confdata is not None:
                for k, v in confdata.items():
                    _confdata[k] = torch.tensor(v, dtype=float_dtype)
                    if _confdata[k].dim() == 1:
                        _confdata[k] = _confdata[k].reshape(-1, 1)
                    _confdata[k] = _confdata[k].transpose(0, 1)  # [?, n_conf, ...]
                    assert _confdata[k].size(1) == _coords.size(
                        1), f'{k} {_confdata[k].size()}, coords {_coords.size()}'

            n_node, n_conf = _coords.shape[0], _coords.shape[1]
            confmask = torch.ones(max_n_confs)  # [n_conf]
            if n_conf >= max_n_confs:
                idx = torch.sort(torch.randperm(n_conf, dtype=torch.int64)[:max_n_confs])[0]
                _coords = _coords[:, idx]
                for k in _confdata:
                    _confdata[k] = _confdata[k][:, idx]
            else:
                # padding confs
                confmask[n_conf:] = 0.
                pad_coords = fake_coords(n_node, max_n_confs - n_conf)
                _coords = torch.concat([_coords, pad_coords], dim=1)  # [n_node, n_conf, 3]
                for k in _confdata:
                    ss = list(_confdata[k].size())
                    pad_values = _confdata[k].new_zeros([ss[0], max_n_confs - n_conf] + ss[2:])
                    _confdata[k] = torch.concat([_confdata[k], pad_values], dim=1)  # [?, n_conf, ...]

            self.confmask = confmask.unsqueeze(0)
            self.coords = _coords
            for k, v in _confdata.items():
                self[k] = v

    def get_count(self, name: str, idx=0, cluster=False):
        counts = self.counts_cluster[:, _count_idx[name]] if cluster else self.counts[:, _count_idx[name]]
        if idx is not None:
            return counts[idx]
        else:
            return counts.clone()

    def set_count(self, name: str, count: int, idx: int = 0, cluster=False):
        if cluster:
            self.counts_cluster[idx, _count_idx[name]] = count
        else:
            self.counts[idx, _count_idx[name]] = count

    @classmethod
    def from_dict(cls, data_dict: dict):
        data = cls()
        for k, v in data_dict.items():
            data[k] = v
        return data

    def to(self, device: str):
        ret = {}
        for k, v in self.items():
            if isinstance(v, Tensor):
                ret[k] = v.to(device)
            else:
                ret[k] = v
        return Data.from_dict(ret)


def get_impropers(graph: MoleculeGraph) -> list[tuple[int]]:
    atomsets = set()
    for atom in graph.get_atoms():
        if atom.atomic_number in [6, 7]:
            idx = atom.idx
            nei = graph.get_neighbor_ids(idx)
            if len(nei) == 3:
                atomsets.add(rkutil.sorted_atomids((idx, nei[0], nei[1], nei[2]), is_improper=True))
    atomsets = list(atomsets)
    atomsets.sort(key=itemgetter(0, 1, 2, 3))
    return atomsets


class GraphData(Data):
    """
    GraphData extract graph and mol features from a molecule.

    Mol features:
    - mol_name: str
    - mapped_smiles: str
    - counts: IntTensor (COUNT_NAMES)  # [1, 9]

    Graph features:
    - node_features: IntTensor (atom_type, connectivity, formal_charge, ring_con, min_ring_size)  # [n_node, 5]
    - edge_features: IntTensor (bond_ring, bond_order)  # [n_edge, 2]
    - proper_mask: FloatTensor  # [n_proper, 1]

    - inc_node_edge: IntTensor  # [n_edge, 2]
    - inc_node_equiv: IntTensor  # [n_node, 1]
    - inc_node_bond: IntTensor  # [n_bond, 2]
    - inc_node_angle: IntTensor  # [n_angle, 3]
    - inc_node_proper: IntTensor  # [n_proper, 4]
    - inc_node_improper: IntTensor  # [n_improper, 4]
    - inc_node_nonbonded14: IntTensor  # [n_nonbonded14, 2]
    - inc_node_nonbonded_all: IntTensor  # [n_nonbonded_all, 2]  if record_nonbonded_all
    - inc_node_mpneighbor: IntTensor  # [n_multipole_frame_neighbors, 2]

    - inc_edge_equiv: IntTensor  # [n_edge, 1]
    - inc_edge_bond: IntTensor  # [n_bond, 1]
    - inc_edge_angle: IntTensor  # [n_angle, 2]
    - inc_edge_proper: IntTensor  # [n_proper, 3]
    - inc_edge_improper: IntTensor  # [n_improper, 3]
    """

    def __init__(self,
                 name: str = '',
                 mapped_smiles: str = '',
                 record_nonbonded_all=True,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32,
                 mol=None,
                 **kwargs):
        super().__init__(int_dtype=int_dtype, float_dtype=float_dtype, **kwargs)

        if not mapped_smiles:
            return

        mol = Molecule.from_mapped_smiles(mapped_smiles, name=name) if mol is None else mol
        pf6_flag = mol.get_smiles() == 'F[P-](F)(F)(F)(F)F'
        if pf6_flag:
            if mol.atomic_numbers[0] == 15:
                pf6_flag = 1
            elif mol.atomic_numbers[1] == 15:
                pf6_flag = 2
            else:
                raise RuntimeError("P should be the first or second atom in PF6")
        self.mol_name = name
        self.mapped_smiles = mol.get_mapped_smiles(isomeric=False)
        self.set_count('mol', 1)

        graph = MoleculeGraph(mol, max_include_ring=MAX_RING_SIZE)
        topos = graph.get_intra_topo()
        topos['ImproperTorsion'] = get_impropers(graph)

        # node features
        atom_type = torch.tensor([ELEMENT_MAP[i] for i in mol.atomic_numbers], dtype=int_dtype)
        connectivity = torch.tensor([atom.connectivity for atom in graph.get_atoms()], dtype=int_dtype)
        formal_charge_vec = torch.tensor(mol.formal_charges, dtype=int_dtype)
        ring_con, min_ring_size = get_ring_info(graph)
        ring_con = torch.tensor(ring_con, dtype=int_dtype)
        min_ring_size = torch.tensor(min_ring_size, dtype=int_dtype)
        features = torch.vstack([atom_type, connectivity, formal_charge_vec, ring_con, min_ring_size]).T
        self.node_features = features  # [n_node, 5]
        self.set_count('node', mol.natoms)
        assert self.get_count('node') == self.node_features.shape[0]

        # edge features
        bond_orders = list(BondOrder)
        edge_idx_dict = {}
        edge_features = []
        for i, atomidx in enumerate(topos['Bond']):
            edge_idx_dict[atomidx] = 2 * i
            edge_idx_dict[atomidx[::-1]] = 2 * i + 1
            bond = graph.get_bond(*atomidx)
            # duplicate for bidirectional edge
            edge_features.append((int(bond.in_ring), bond_orders.index(BondOrder(bond.order))))
            edge_features.append((int(bond.in_ring), bond_orders.index(BondOrder(bond.order))))
        self.edge_features = torch.tensor(edge_features, dtype=int_dtype).reshape(-1, 2)

        # topological indecies
        for term, width in MMTERM_WIDTH.items():
            atomidxs = topos[MM_TOPO_MAP[term]]
            if pf6_flag and term is MMTerm.angle:
                if pf6_flag == 1:
                    atomidxs = [(1, 0, 2), (1, 0, 3), (1, 0, 5), (1, 0, 6), (2, 0, 3), (2, 0, 4), (2, 0, 6), (3, 0, 4),
                                (3, 0, 5), (4, 0, 5), (4, 0, 6), (5, 0, 6)]
                else:
                    atomidxs = [(0, 1, 2), (0, 1, 3), (0, 1, 5), (0, 1, 6), (2, 1, 3), (2, 1, 4), (2, 1, 6), (3, 1, 4),
                                (3, 1, 5), (4, 1, 5), (4, 1, 6), (5, 1, 6)]
            index = torch.tensor(atomidxs, dtype=int_dtype).reshape(-1, width)
            self[f'inc_node_{term.name}'] = index
            self.set_count(term.name, len(atomidxs))
            edge_indices = []
            for ids in atomidxs:
                eids = []
                for i in range(len(ids) - 1):
                    if term is not MMTerm.improper:
                        eid = edge_idx_dict[(ids[i], ids[i + 1])]
                    else:
                        eid = edge_idx_dict[((ids[0], ids[i + 1]))]
                    eids.append(eid)
                edge_indices.append(eids)
            self[f'inc_edge_{term.name}'] = torch.tensor(edge_indices, dtype=int_dtype).reshape(-1, width - 1)
        bond_idx = self.inc_node_bond
        self.inc_node_edge = torch.concat([bond_idx, bond_idx.flip(-1)], dim=-1).reshape(-1, 2)
        self.set_count('edge', self.inc_node_edge.shape[0])

        self.pf6_bond_mask = torch.ones(6, dtype=float_dtype) if pf6_flag else torch.zeros(self.inc_node_bond.shape[0],
                                                                                           dtype=float_dtype)
        self.pf6_angle_mask = torch.ones(12, dtype=float_dtype) if pf6_flag else torch.zeros(
            self.inc_node_angle.shape[0], dtype=float_dtype)

        # equivalent
        atom_equi_index, edge_equi_index = find_equivalent_index(mol, self.inc_node_edge.tolist())
        self.inc_node_equiv = torch.tensor(atom_equi_index, dtype=int_dtype).reshape(-1)
        self.inc_edge_equiv = torch.tensor(edge_equi_index, dtype=int_dtype).reshape(-1)

        nb12 = graph.topo.nonbonded12_pairs
        nb13 = graph.topo.nonbonded13_pairs
        nb14 = graph.topo.nonbonded14_pairs
        nb15 = graph.topo.nonbonded15_pairs
        self.inc_node_nonbonded12 = torch.tensor(nb12, dtype=int_dtype).reshape(-1, 2)
        self.set_count('nonbonded12', self.inc_node_nonbonded12.shape[0])
        self.inc_node_nonbonded13 = torch.tensor(nb13, dtype=int_dtype).reshape(-1, 2)
        self.set_count('nonbonded13', self.inc_node_nonbonded13.shape[0])
        self.inc_node_nonbonded14 = torch.tensor(nb14, dtype=int_dtype).reshape(-1, 2)
        self.set_count('nonbonded14', self.inc_node_nonbonded14.shape[0])
        self.inc_node_nonbonded15 = torch.tensor(nb15, dtype=int_dtype).reshape(-1, 2)
        self.set_count('nonbonded15', self.inc_node_nonbonded15.shape[0])

        # nonbonded all
        if record_nonbonded_all:
            nball = graph.topo.nonbondedall_pairs
            self.inc_node_nonbonded_all = torch.tensor(nball, dtype=int_dtype).reshape(-1, 2)
            self.set_count('nonbonded_all', self.inc_node_nonbonded_all.shape[0])

        # mask linear proper
        matches = match_linear_proper(mol)
        mask = torch.ones(self.get_count('proper'), dtype=float_dtype)
        for i, atomidx in enumerate(self.inc_node_proper):
            at = tuple(atomidx.tolist())
            if at in matches:
                mask[i] = 0.
        self.proper_mask = mask


class MonoData(GraphData):
    """"
    Data containing one molecule

    3D features & labels: 
    - coords: FloatTensor  # [n_node, n_confs, 3]
    - forces: FloatTensor  # [n_node, n_confs, 3]
    - energy: FloatTensor  # [1, n_conf]

    - inc_node_edge3d: IntFloat  # [n_edge3d, 2]
    - conf_mask: FloatTensor  # [1, n_conf]
    """

    @staticmethod
    def get_edge_3d(coords: Tensor, exists_edges: Tensor, cutoff: float):

        exists_edges = exists_edges.tolist()
        exists_edges = set((tuple(p) for p in exists_edges))

        coords0 = coords.unsqueeze(0).repeat((coords.shape[0], 1, 1, 1))
        coords1 = coords0.transpose(0, 1)  # [n_node, n_node, n_conf, 3]
        r2 = torch.min(torch.sum((coords1 - coords0)**2, dim=-1), dim=-1)[0]
        ii, jj = torch.where(r2 <= cutoff**2)

        edge3d = []
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i == j or (i, j) in exists_edges:
                continue
            edge3d.append([i, j])
        return edge3d

    def set_edge_3d(self, edge3d_rcut=5.):
        coords = self.coords[:, self.confmask[0] > 0.]
        edge3d = self.get_edge_3d(coords, self.inc_node_edge, edge3d_rcut)
        self.inc_node_edge3d = torch.tensor(edge3d, dtype=self.inc_node_edge.dtype).reshape(-1, 2)

    def __init__(self,
                 name: str = '',
                 mapped_smiles: str = '',
                 moldata: dict[str, np.ndarray] = None,
                 confdata: dict[str, np.ndarray] = None,
                 max_n_confs: int = 10,
                 edge3d_rcut=5.,
                 record_nonbonded_all=True,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32,
                 check_bond=False,
                 mol=None):
        super().__init__(name=name,
                         mapped_smiles=mapped_smiles,
                         record_nonbonded_all=record_nonbonded_all,
                         int_dtype=int_dtype,
                         float_dtype=float_dtype,
                         moldata=moldata,
                         confdata=confdata,
                         max_n_confs=max_n_confs,
                         mol=mol)

        if not mapped_smiles:
            return

        if check_bond:
            mol = Molecule.from_mapped_smiles(self.mapped_smiles) if mol is None else mol
            mol.append_conformers(Conformer(np.random.rand(mol.natoms, 3), mol.atomic_symbols))
            for iconf in range(self.confmask.size(1)):
                if self.confmask[0, iconf] > 1e-2:
                    mol.conformers[0].coords = self.coords[:, iconf].numpy()
                    if check_new_bonds(mol) or check_broken_bonds(mol):
                        self.confmask[0, iconf] = 0.

        if 'coords' in self:
            self.set_edge_3d(edge3d_rcut=edge3d_rcut)


def collate_data(data_list: list[Data]):
    """
    Collate a list of data, according to the increasing rule.
    """

    def collate_tensor(k: str, vs: list[Tensor], incs: dict[str, Tensor], cluster: bool):
        cat_vs = torch.concat(vs, dim=0)
        if k.startswith('inc_'):
            inc_name = k.split('_')[1]
            if inc_name in incs:
                inc = incs[inc_name]
            else:
                inc = torch.tensor([data.get_count(inc_name, cluster=cluster) for data in data_list],
                                   device=vs[0].device,
                                   dtype=vs[0].dtype)
                inc = torch.concat(
                    (torch.tensor([0], device=vs[0].device, dtype=vs[0].dtype), torch.cumsum(inc, 0)[:-1]), dim=0)
                incs[inc_name] = inc
            nums = torch.tensor([v.shape[0] for v in vs], dtype=vs[0].dtype, device=vs[0].device)
            size = (-1,) + (1,) * (vs[0].dim() - 1)
            cat_vs += torch.repeat_interleave(inc, nums).view(size)
        return cat_vs

    cat_data = Data()
    data0 = data_list[0]
    cluster = 'inc_node_nonbonded_all_cluster' in data0
    incs = {}
    for k in data0.keys():
        if k == 'pf6_bond_mask' and (not all([k in data for data in data_list])):
            continue
        if k == 'pf6_angle_mask' and (not all([k in data for data in data_list])):
            continue
        vs = [data[k] for data in data_list]
        if isinstance(vs[0], Tensor):
            vs = collate_tensor(k, vs, incs, cluster)
        cat_data[k] = vs
    return cat_data


class ClusterData(Data):
    """"
    Data containing multiple molecules

    3D features & labels: 
    - coords: FloatTensor  # [n_node, n_confs, 3]
    - forces: FloatTensor  # [n_node, n_confs, 3]
    - energy: FloatTensor  # [1, n_conf]

    - inc_node_edge3d: IntFloat  # [n_edge3d, 2]
    - conf_mask: FloatTensor  # [1, n_conf]
    """

    def init_sub_graph(self, max_n_confs, record_nonbonded_all, int_dtype, float_dtype, mapped_smiles, **kwargs):
        coords = self.coords.transpose(0, 1).numpy() if 'coords' in self else None
        shift = 0
        graph_list = []
        for i, mps in enumerate(mapped_smiles):
            mol = Molecule.from_mapped_smiles(mps)
            na = mol.natoms
            confdata = {'coords': coords[:, shift:shift + na]} if coords is not None else None
            graph = MonoData(str(i),
                             mps,
                             confdata=confdata,
                             record_nonbonded_all=record_nonbonded_all,
                             max_n_confs=max_n_confs,
                             int_dtype=int_dtype,
                             float_dtype=float_dtype,
                             mol=mol,
                             **kwargs)
            shift += na
            graph_list.append(graph)

        return graph_list

    def __init__(self,
                 name: str = None,
                 mapped_smiles: list[str] = None,
                 moldata: dict[str, np.ndarray] = None,
                 confdata: dict[str, np.ndarray] = None,
                 max_n_confs=10,
                 edge3d_rcut=5.,
                 nb_cutoff=None,
                 record_nonbonded_all=True,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32,
                 **kwargs):
        super().__init__(int_dtype=int_dtype,
                         float_dtype=float_dtype,
                         moldata=moldata,
                         confdata=confdata,
                         max_n_confs=max_n_confs)

        self.name = name

        if not mapped_smiles:
            return

        graph_list = self.init_sub_graph(max_n_confs, record_nonbonded_all, int_dtype, float_dtype, mapped_smiles,
                                         **kwargs)
        graphs = collate_data(graph_list)
        for k, v in graphs.items():
            if k == 'confmask':
                continue
            self[k] = v
        natoms = self.get_count('node', idx=None).tolist()
        nonbonded_all_inter = []
        if nb_cutoff is None:
            for imol, n in enumerate(natoms[:-1]):
                shift_i = sum(natoms[:imol])
                shift_j = sum(natoms[:imol + 1])
                for i in range(n):
                    for j in range(sum(natoms[imol + 1:])):
                        nonbonded_all_inter.append([i + shift_i, j + shift_j])
        else:
            assert max_n_confs == 1
            coords = self.coords[:, 0]
            cutoff2 = nb_cutoff**2
            cutoff2_l = (nb_cutoff + 10.)**2
            mask = (coords.unsqueeze(0) - coords.unsqueeze(1)).square().sum(dim=-1) < cutoff2
            for imol, n in enumerate(natoms[:-1]):
                shift_i = sum(natoms[:imol])
                cc_i = coords[shift_i:shift_i + natoms[imol]].mean(dim=0)
                for jmol in range(imol + 1, len(natoms)):
                    shift_j = sum(natoms[:jmol])
                    cc_j = coords[shift_j:shift_j + natoms[jmol]].mean(dim=0)
                    d2 = (cc_i - cc_j).square().sum()
                    if d2 > cutoff2_l:
                        continue
                    for i in range(n):
                        for j in range(natoms[jmol]):
                            ii = i + shift_i
                            jj = j + shift_j
                            if mask[ii, jj]:
                                nonbonded_all_inter.append([ii, jj])

        nonbonded_all_inter = torch.tensor(nonbonded_all_inter, dtype=int_dtype).reshape(-1, 2)
        self.inc_node_nonbonded_all_cluster = torch.concat((self.inc_node_nonbonded_all, nonbonded_all_inter), dim=0)
        self.counts_cluster = torch.sum(self.counts, dim=0, keepdim=True)
        self.set_count('nonbonded_all', self.inc_node_nonbonded_all_cluster.shape[0], cluster=True)

        if hasattr(self, 'confmask'):
            self.confmask_cluster = self.confmask.clone().detach()
            self.confmask = self.confmask.repeat(self.get_count('mol', cluster=True), 1)
