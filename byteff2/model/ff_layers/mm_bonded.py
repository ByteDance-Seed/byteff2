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

from functools import partial
from typing import Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

from byteff2.data.data import ClusterData, MonoData

from .base import FFLayer, PreFFLayer
from .utils import get_angle_vec, get_dihedral_angle_vec, get_distance_vec, reduce_counts, set_grad_max

PROPERTORSION_TERMS = 4


class CustomClamp(torch.autograd.Function):  # pylint: disable=abstract-method

    @staticmethod
    def forward(ctx, _input, _min=None, _max=None):
        return _input.clamp(min=_min, max=_max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


custom_clamp = CustomClamp.apply

term_shapes = {'bond': 2, 'angle': 3, 'proper': 4, 'improper': 2}


class PreMMBonded(PreFFLayer):

    term_param_map = {
        'bond': {
            'bond_k': 1,
            'bond_r0': 1
        },
        'angle': {
            'angle_k': 1,
            'angle_d0': 1
        },
        'proper': {
            'proper_k': PROPERTORSION_TERMS,
            # 'proper_d0': PROPERTORSION_TERMS
        },
        'improper': {
            'improper_k': 1,
            # 'improper_d0': 1  # fix to pi
        }
    }

    param_std_mean_range = {
        'bond_k': (200., 700., 80., 4000.),
        'bond_r0': (0.19, 1.3, 0.5, 5.),
        'angle_k': (60., 130., 40., 1000.),
        'angle_d0': (10., 120., 20., 180.),
        'proper_k': (1.2, 0.4, -20., 20.),
        'improper_k': (3.7, 4.3, 0., 20.)
    }

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            pre_mlp_dims=(32, 32, 3),  # (hidden, out, layers)
            post_mlp_dims=(32, 32, 3),  # (hidden, out, layers)
            out_mlp_dims=(32, 3),  # (hidden, layers)
            act='gelu',
            tanh_output=15.,
            grad_max=None,
            **configs):
        super().__init__(node_dim, edge_dim)

        self.tanh_output = tanh_output
        self.grad_max = grad_max
        self.pre_mlp: dict[str, MLP] = nn.ModuleDict()
        self.post_mlp: dict[str, MLP] = nn.ModuleDict()
        self.out_mlp: dict[str, MLP] = nn.ModuleDict()

        for term, shape in term_shapes.items():
            self.pre_mlp[term] = MLP(in_channels=shape * node_dim + (shape - 1) * edge_dim,
                                     hidden_channels=pre_mlp_dims[0],
                                     out_channels=pre_mlp_dims[1],
                                     num_layers=pre_mlp_dims[2],
                                     norm=None,
                                     act=act)
            self.post_mlp[term] = MLP(in_channels=pre_mlp_dims[1],
                                      hidden_channels=pre_mlp_dims[0],
                                      out_channels=pre_mlp_dims[1],
                                      num_layers=pre_mlp_dims[2],
                                      norm=None,
                                      act=act)
        for term, params in self.term_param_map.items():
            for p, w in params.items():
                self.out_mlp[p] = MLP(in_channels=post_mlp_dims[1],
                                      hidden_channels=out_mlp_dims[0],
                                      out_channels=w,
                                      num_layers=out_mlp_dims[1],
                                      norm=None,
                                      act=act)

    def reset_parameters(self):
        for module in self.pre_mlp.values():
            module.reset_parameters()
        for module in self.post_mlp.values():
            module.reset_parameters()
        for module in self.out_mlp.values():
            module.reset_parameters()

    def _symmetric_pooling(self, x_h: Tensor, e_h: Tensor, graph: MonoData) -> dict[str, Tensor]:
        '''post_mlp -> symmetry-preserving pooling with bond -> post_mlp -> out_mlp'''
        ret = {}
        for term in term_shapes:
            node_idx = graph[f'inc_node_{term}'].long()
            edge_idx = graph[f'inc_edge_{term}'].long()
            width = node_idx.shape[1]

            xs = []
            for i in range(width):
                xs.append(x_h[node_idx[:, i]])
                if i < width - 1:
                    xs.append(e_h[edge_idx[:, i]])

            if term != 'improper':
                xs = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))
            else:
                xs = (
                    torch.concat([xs[0], xs[1], xs[2]], dim=-1),
                    torch.concat([xs[0], xs[3], xs[4]], dim=-1),
                    torch.concat([xs[0], xs[5], xs[6]], dim=-1),
                )

            y = sum([self.pre_mlp[term](x) for x in xs])
            y = self.post_mlp[term](y)

            for param in self.term_param_map[term]:
                p = self.out_mlp[param](y)
                ret[param] = p
        return ret

    def set_grad_max(self, params: dict[str, Tensor]):
        new_params = {}
        if self.grad_max is not None:
            for k, v in params.items():
                p = k.split('.')[1]
                if p in self.grad_max:
                    v = v.clone()
                    v.register_hook(partial(set_grad_max, max_step=self.grad_max[p]))
                v.retain_grad()
                new_params[k] = v
        else:
            new_params = params
        return new_params

    def patch_pf6(self, data: MonoData, param: Tensor, term: str):
        if 'pf6_bond_mask' not in data:
            return param

        bk, bl = 1000., 1.65
        ak, al = 278.44, 90. / 180. * torch.pi

        bk2 = (bl - PreMMBondedConj.bond_b1) / (PreMMBondedConj.bond_b2 - PreMMBondedConj.bond_b1) * bk
        bk1 = bk - bk2

        ak2 = (al - PreMMBondedConj.angle_b1) / (PreMMBondedConj.angle_b2 - PreMMBondedConj.angle_b1) * ak
        ak1 = ak - ak2

        if term == 'bond_k1':
            d = torch.where(data.pf6_bond_mask.unsqueeze(-1) > 0.9, bk1, param)
        elif term == 'bond_k2':
            d = torch.where(data.pf6_bond_mask.unsqueeze(-1) > 0.9, bk2, param)
        elif term == 'angle_k1':
            d = torch.where(data.pf6_angle_mask.unsqueeze(-1) > 0.9, ak1, param)
        elif term == 'angle_k2':
            d = torch.where(data.pf6_angle_mask.unsqueeze(-1) > 0.9, ak2, param)
        else:
            d = param
        return d

    def forward(self,
                data: MonoData,
                x_h: Tensor,
                e_h: Tensor,
                ff_parameters: dict[str, Tensor] = None) -> dict[str, Tensor]:
        params = self._symmetric_pooling(x_h, e_h, data)
        ff_parameters = {}
        for term in params:
            d = params[term]
            if self.tanh_output > 0.:
                d = self.tanh_output * torch.tanh(d)
            const = self.param_std_mean_range[term]
            d = d * const[0] + const[1]
            d = self.patch_pf6(data, d, term)
            ff_parameters[f'{type(self).__name__}.{term}'] = custom_clamp(d, const[2], const[3])

        if 'proper_mask' in data:
            ff_parameters[f'{type(self).__name__}.proper_k'] *= data.proper_mask.unsqueeze(-1)
        ff_parameters = self.set_grad_max(ff_parameters)
        return ff_parameters


class MMBonded(FFLayer):

    def reset_parameters(self):
        pass

    def calc_bond(self, graph: MonoData, ff_params: dict, cluster=False):
        k = ff_params[f'Pre{type(self).__name__}.bond_k']
        r0 = ff_params[f'Pre{type(self).__name__}.bond_r0']
        coords = graph.coords
        n_conf = coords.shape[1]
        node_idx = graph.inc_node_bond.long()
        counts = graph.get_count('bond', idx=None, cluster=cluster)
        if node_idx.shape[0] == 0:
            return 0., 0.

        cc = [coords[node_idx[:, i]] for i in range(node_idx.shape[1])]
        r12, r12vec = get_distance_vec(*cc)

        pair_energy = 0.5 * k * (r12 - r0)**2  # [nbonds, n_conf]
        energy = reduce_counts(pair_energy, counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(coords)  # [n_atom, n_conf, 3]
        pair_forces = (k * (1 - r0 / r12)).unsqueeze(-1) * r12vec  # [nbonds, n_conf, 3]
        atom1_idxs = node_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = node_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        # IMPORTANT: negative sign
        forces.scatter_add_(0, atom1_idxs, pair_forces)
        forces.scatter_add_(0, atom2_idxs, -pair_forces)
        return energy, forces

    def calc_angle(self, graph: MonoData, ff_params: dict, cluster=False):
        k = ff_params[f'Pre{type(self).__name__}.angle_k']
        d0 = torch.deg2rad(ff_params[f'Pre{type(self).__name__}.angle_d0'])
        coords = graph.coords
        n_conf = coords.shape[1]
        node_idx = graph.inc_node_angle.long()
        counts = graph.get_count('angle', idx=None, cluster=cluster)

        if node_idx.shape[0] == 0:
            return 0., 0.

        cc = [coords[node_idx[:, i]] for i in range(node_idx.shape[1])]
        theta, f1, f3 = get_angle_vec(*cc)
        pair_energy = 0.5 * k * (theta - d0)**2  # [nangles, n_conf]
        energy = reduce_counts(pair_energy, counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(coords)  # [n_atom, n_conf, 3]
        # [nbatch, n_conf, nangles]
        atom1_idxs = node_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = node_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom3_idxs = node_idx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        fc = -(k * (theta - d0)).unsqueeze(-1)  # [nangles, n_conf, 1]
        force1 = fc * f1
        force3 = fc * f3
        force2 = -force1 - force3
        forces.scatter_add_(0, atom1_idxs, force1)
        forces.scatter_add_(0, atom2_idxs, force2)
        forces.scatter_add_(0, atom3_idxs, force3)

        return energy, forces

    def _calc_dihedral_energy_forces(cls, coords, node_idx, counts, k, periodicity, phase) -> tuple[Tensor, Tensor]:
        n_conf = coords.shape[1]
        n_term = k.shape[-1]
        k = k.unsqueeze(1).expand(-1, n_conf, -1)  # [ndihedrals, n_conf, nterms]
        periodicity = periodicity.unsqueeze(1).expand(-1, n_conf, -1)  # [ndihedrals, n_conf, nterms]
        phase = phase.unsqueeze(1).expand(-1, n_conf, -1)  # [ndihedrals, n_conf, nterms]

        cc = [coords[node_idx[:, i]] for i in range(node_idx.shape[1])]
        theta, f1, f2, f3, f4 = get_dihedral_angle_vec(*cc)
        theta_expanded = theta.unsqueeze(-1).expand(-1, -1, n_term)  # [ndihedrals, n_conf, nterms]
        dtheta = periodicity * theta_expanded - phase  # [ndihedrals, n_conf, nterms]
        # [ndihedrals, n_conf]
        dihedral_energy = torch.sum(k * (1 + torch.cos(dtheta)), dim=-1)
        energy = reduce_counts(dihedral_energy, counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(coords)  # [n_atom, n_conf, 3]

        # [ndihedrals, n_conf, 1]
        force_prefac = torch.sum(k * periodicity * torch.sin(dtheta), dim=-1).unsqueeze(-1)
        force1 = force_prefac * f1  # [ndihedrals, n_conf, 3]
        force2 = force_prefac * f2  # [ndihedrals, n_conf, 3]
        force3 = force_prefac * f3  # [ndihedrals, n_conf, 3]
        force4 = force_prefac * f4  # [ndihedrals, n_conf, 3]

        # [ndihedrals, n_conf, 3]
        atom1_idxs = node_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = node_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom3_idxs = node_idx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom4_idxs = node_idx[:, 3].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)

        # [ndihedrals, n_conf, 3] -> [n_atom, n_conf, 3]
        forces.scatter_add_(0, atom1_idxs, force1)
        forces.scatter_add_(0, atom2_idxs, force2)
        forces.scatter_add_(0, atom3_idxs, force3)
        forces.scatter_add_(0, atom4_idxs, force4)

        return energy, forces

    def calc_proper(self, graph: MonoData, ff_params: dict, cluster=False):
        k = ff_params[f'Pre{type(self).__name__}.proper_k']
        proper_n = [n + 1 for n in range(PROPERTORSION_TERMS)]
        periodicity = torch.tensor(proper_n, dtype=k.dtype, device=k.device).unsqueeze(0).expand(k.shape[0], -1)
        phase = (((torch.tensor(proper_n, device=k.device, dtype=torch.int64) + 1) % 2) * torch.pi).to(
            k.dtype).unsqueeze(0).expand(k.shape[0], -1)
        node_idx = graph.inc_node_proper.long()
        counts = graph.get_count('proper', idx=None, cluster=cluster)

        if node_idx.shape[0] == 0:
            return 0., 0.

        return self._calc_dihedral_energy_forces(graph.coords, node_idx, counts, k, periodicity, phase)

    def calc_improper(self, graph: MonoData, ff_params: dict, cluster=False):
        k = ff_params[f'Pre{type(self).__name__}.improper_k']
        periodicity = torch.ones_like(k) * 2.
        phase = torch.ones_like(k) * torch.pi
        node_idx = graph.inc_node_improper.long()
        counts = graph.get_count('improper', idx=None, cluster=cluster)
        if node_idx.shape[0] == 0:
            return 0., 0.

        return self._calc_dihedral_energy_forces(graph.coords, node_idx, counts, k, periodicity, phase)

    def forward(self,
                data: Union[MonoData, ClusterData],
                x_h: Tensor,
                e_h: Tensor,
                ff_parameters: dict[str, Tensor],
                cluster: bool = False):

        energy, forces = 0., 0.
        for term in term_shapes:
            e_term, f_term = getattr(self, f"calc_{term}")(data, ff_parameters, cluster=cluster)
            energy += e_term
            forces += f_term
            ff_parameters[f'{type(self).__name__}.{term}_energy'] = e_term
            ff_parameters[f'{type(self).__name__}.{term}_forces'] = f_term

        confmask = data.confmask_cluster if cluster else data.confmask
        n_atom = data.get_count('node', idx=None, cluster=cluster)
        confmask_forces = confmask.repeat_interleave(n_atom, 0).unsqueeze(-1).repeat(1, 1, 3)
        energy *= confmask
        forces *= confmask_forces

        return energy, forces


class PreMMBondedConj(PreMMBonded):
    term_param_map = {
        'bond': {
            'bond_k1': 1,
            'bond_k2': 1
        },
        'angle': {
            'angle_k1': 1,
            'angle_k2': 1
        },
        'proper': {
            'proper_k': PROPERTORSION_TERMS,
            # 'proper_d0': PROPERTORSION_TERMS
        },
        'improper': {
            'improper_k': 1,
            # 'improper_d0': 1  # fix to pi
        }
    }
    param_std_mean_range = {
        'bond_k1': (50., 1000., 0., 4000.),
        'bond_k2': (25., 500., 0., 4000.),
        'angle_k1': (5., 100., 0., 400.),
        'angle_k2': (5., 100., 0., 400.),
        'proper_k': (1.2, 0.4, -20., 20.),
        'improper_k': (3.7, 4.3, 0., 40.)
    }

    bond_b1 = 0.5
    bond_b2 = 4.0
    angle_b1 = 0.25 * torch.pi
    angle_b2 = 1.05 * torch.pi


class MMBondedConj(MMBonded):

    def calc_bond(self, graph: MonoData, ff_params: dict, cluster=False):
        k1 = ff_params[f'Pre{type(self).__name__}.bond_k1']
        k2 = ff_params[f'Pre{type(self).__name__}.bond_k2']
        b1, b2 = PreMMBondedConj.bond_b1, PreMMBondedConj.bond_b2
        coords = graph.coords
        n_conf = coords.shape[1]
        node_idx = graph.inc_node_bond.long()
        counts = graph.get_count('bond', idx=None, cluster=cluster)

        if node_idx.shape[0] == 0:
            return 0., 0.

        cc = [coords[node_idx[:, i]] for i in range(node_idx.shape[1])]
        r12, r12vec = get_distance_vec(*cc)

        pair_energy = 0.5 * k1 * (r12 - b1)**2 + 0.5 * k2 * (r12 - b2)**2  # [nbonds, n_conf]
        pair_energy += 0.5 * (-k1 * b1**2 - k2 * b2**2 + (k1 * b1 + k2 * b2)**2 / (k1 + k2))
        energy = reduce_counts(pair_energy, counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(coords)  # [n_atom, n_conf, 3]
        pair_forces = (k1 * (1 - b1 / r12) + k2 * (1 - b2 / r12)).unsqueeze(-1) * r12vec  # [nbonds, n_conf, 3]
        atom1_idxs = node_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = node_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        # IMPORTANT: negative sign
        forces.scatter_add_(0, atom1_idxs, pair_forces)
        forces.scatter_add_(0, atom2_idxs, -pair_forces)
        return energy, forces

    def calc_angle(self, graph: MonoData, ff_params: dict, cluster=False):
        k1 = ff_params[f'Pre{type(self).__name__}.angle_k1']
        k2 = ff_params[f'Pre{type(self).__name__}.angle_k2']
        b1, b2 = PreMMBondedConj.angle_b1, PreMMBondedConj.angle_b2
        coords = graph.coords
        n_conf = coords.shape[1]
        node_idx = graph.inc_node_angle.long()
        counts = graph.get_count('angle', idx=None, cluster=cluster)

        if node_idx.shape[0] == 0:
            return 0., 0.

        cc = [coords[node_idx[:, i]] for i in range(node_idx.shape[1])]
        theta, f1, f3 = get_angle_vec(*cc)
        pair_energy = 0.5 * k1 * (theta - b1)**2 + 0.5 * k2 * (theta - b2)**2  # [nangles, n_conf]
        pair_energy += 0.5 * (-k1 * b1**2 - k2 * b2**2 + (k1 * b1 + k2 * b2)**2 / (k1 + k2))
        energy = reduce_counts(pair_energy, counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(coords)  # [n_atom, n_conf, 3]
        # [nbatch, n_conf, nangles]
        atom1_idxs = node_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = node_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom3_idxs = node_idx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        fc = -(k1 * (theta - b1) + k2 * (theta - b2)).unsqueeze(-1)  # [nangles, n_conf, 1]
        force1 = fc * f1
        force3 = fc * f3
        force2 = -force1 - force3
        forces.scatter_add_(0, atom1_idxs, force1)
        forces.scatter_add_(0, atom2_idxs, force2)
        forces.scatter_add_(0, atom3_idxs, force3)

        return energy, forces
