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
from typing import Union

import torch
from torch import LongTensor, Tensor
from torch.nn import Embedding
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

from byteff2.data.data import ClusterData, MonoData
from byteff2.utils.definitions import (ALPHA_FREE, C6_FREE, CHG_FACTOR, ELEMENT_MAP, RVDW_FREE, SUPPORTED_ELEMENTS,
                                       V_FREE)

from .base import FFLayer, PreFFLayer
from .utils import get_distance_vec, reduce_counts, to_dense_batch, to_dense_index

logger = logging.getLogger(__name__)


class PreChargeVolume(PreFFLayer):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            pre_mlp_dims=(32, 32, 3),  # (hidden, out, layers)
            out_mlp_dims=(32, 3),  # (hidden, layers)
            act='gelu',
            **configs):
        super().__init__(node_dim, edge_dim)

        self.charge_range = 4.
        self.Li_volume = 13.
        self.volume_mlp = MLP(in_channels=node_dim,
                              hidden_channels=out_mlp_dims[0],
                              out_channels=1,
                              num_layers=out_mlp_dims[1],
                              norm=None,
                              act=act)
        self.charge_pre_mlp = MLP(in_channels=node_dim * 2 + edge_dim,
                                  hidden_channels=pre_mlp_dims[0],
                                  out_channels=pre_mlp_dims[1],
                                  num_layers=pre_mlp_dims[2],
                                  norm=None,
                                  act=act)
        self.charge_out_mlp = MLP(in_channels=pre_mlp_dims[1],
                                  hidden_channels=out_mlp_dims[0],
                                  out_channels=1,
                                  bias=False,
                                  num_layers=out_mlp_dims[1],
                                  norm=None,
                                  act='tanh')

    def reset_parameters(self):
        self.volume_mlp.reset_parameters()
        self.charge_pre_mlp.reset_parameters()
        self.charge_out_mlp.reset_parameters()

    def _bcc_charge(self, graph: MonoData, x_h: Tensor, e_h: Tensor):
        node_idx = graph['inc_node_bond'].long()
        edge_idx = graph['inc_edge_bond'].long()
        xs = [x_h[node_idx[:, 0]], e_h[edge_idx[:, 0]], x_h[node_idx[:, 1]]]
        xs = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))
        y0 = self.charge_pre_mlp(xs[0])
        y1 = self.charge_pre_mlp(xs[1])
        bcc = self.charge_out_mlp(y0 - y1).squeeze(-1)
        bcc = torch.tanh(bcc) * self.charge_range

        # formal charge
        charge = graph.node_features[:, 2].clone().to(bcc.dtype)
        # average symmetric atoms
        equiv_idx = graph.inc_node_equiv.long()
        charge = scatter(charge, equiv_idx, 0, reduce='mean')[equiv_idx]
        # add bcc
        charge.scatter_add_(0, node_idx[:, 0], bcc)
        charge.scatter_add_(0, node_idx[:, 1], -bcc)
        return charge.unsqueeze(-1)

    def forward(self,
                data: MonoData,
                x_h: Tensor,
                e_h: Tensor,
                ff_parameters: dict[str, Tensor] = None) -> dict[str, Tensor]:
        if ff_parameters is None:
            ff_parameters = {}
        charge = self._bcc_charge(data, x_h, e_h)
        ff_parameters['PreChargeVolume.charges'] = charge

        # this v_free is written in definition, different from the trainable hyper parameter
        v_free = x_h.new(V_FREE)
        atomic_number = data.node_features[:, 0].long()
        v_free = v_free[atomic_number]

        # predict relative volume
        volume_ratiao = torch.exp(self.volume_mlp(x_h)).squeeze(-1)
        volume = volume_ratiao * v_free
        volume = torch.where(atomic_number == ELEMENT_MAP[3], self.Li_volume, volume)  # fix volume of Li
        ff_parameters['PreChargeVolume.volumes'] = volume.unsqueeze(-1)
        return ff_parameters


class ChargeVolume(FFLayer):

    def reset_parameters(self):
        pass

    def forward(self, data, x_h, e_h, ff_parameters, cluster=False):
        return 0., 0.


class PolarizationSolver(torch.nn.Module):

    def __init__(self, a=0.39):
        super(PolarizationSolver, self).__init__()
        self.a = a
        logger.info(f'use a={self.a} for polarization solver')

    def compute_polarizability_tensor(self, n_batch, n_conf, polarizability):
        """
        Compute the tensor alpha which is a 3N x 3N matrix, 
        each 3 x 3 submatrix is a diagnoal matrix with diagonal value being an isotropic polarizability value
        """
        polar_matrix = torch.diag_embed((1 / polarizability).repeat_interleave(3, dim=-1))
        polar_matrix = polar_matrix.unsqueeze(1).expand(n_batch, n_conf, -1, -1)

        return polar_matrix

    def compute_damping_tensor(self, r_norm, polarizability_interaction):
        """
        Compute the damping tensor which is Sij * (1 - exp(-au^3))
        Uij = rij / (ai * aj) ** (1/6)
        
        return a torch tensor of (N, N, n)
        """
        damp = -self.a * r_norm.pow(3) / polarizability_interaction.pow(0.5)
        damp = torch.clip(damp, min=-1e6)

        exp_damp = torch.exp(damp)
        D1 = 1 - exp_damp
        D2 = 1 - (1 - damp) * exp_damp
        D2 = torch.clip(D2, min=1e-6)
        return D1, D2

    def compute_electric_field(
        self,
        r_ij,
        r_norm,
        D1,
        scaler,
        charge,
    ):
        """
        charge: [n_batch, n_atom]
        D1: [n_batch, n_conf, n_atom, n_atom]
        scaler: [n_batch, n_conf, n_atom, n_atom]
        """
        coeff_p = D1 * scaler
        coeff_p = coeff_p.unsqueeze(-1).expand(-1, -1, -1, -1, 3)  # [n_batch, n_conf, n_atom, n_atom, 3]

        r_norm = r_norm.unsqueeze(-1).expand(-1, -1, -1, -1, 3)  # [n_batch, n_conf, n_atom, n_atom, 3]
        p_vec = coeff_p * r_ij / r_norm.pow(3)

        E_p = p_vec * charge.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        E_p = E_p.sum(dim=-2)
        n_batch, n_conf, n_atom, _ = E_p.shape
        E_p = E_p.reshape(n_batch, n_conf, n_atom * 3)

        return E_p

    def compute_interaction_tensor(self, r_ij, r_norm, D1, D2, node_mask):
        """
        Compute the tensor T_ij within a 3N x 3N supermatrix for an array of atomic positions,
        with self-interactions (i=j) set to zero to avoid singularities.
        
        Returns:
            torch.Tensor: A tensor of shape (3N, 3N) containing the supermatrix with each 3x3 block T_ij.
        """

        n_batch, n_conf, n_atom, _, _ = r_ij.shape
        eye = torch.eye(3, device=r_ij.device, dtype=r_ij.dtype).expand(n_batch, n_conf, n_atom, n_atom, 3, 3)

        # Outer product of r_ij with itself
        r_norm = r_norm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3)

        D1 = D1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3)
        D2 = D2.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3)

        rij_outer = torch.einsum('...i,...j->...ij', r_ij, r_ij)

        # Compute T_ij for all pairs
        T_ij = -(3 * rij_outer * D2 - r_norm.pow(2) * eye * D1) / r_norm.pow(5)

        # Flatten T_ij into a 3N x 3N matrix
        T_supermatrix = T_ij.transpose(-2, -3).reshape(n_batch, n_conf, n_atom * 3, n_atom * 3)

        # mask fake atoms
        mask = node_mask.unsqueeze(1).unsqueeze(-1).expand(-1, n_conf, -1, 3)
        mask = mask.reshape(n_batch, n_conf, n_atom * 3).to(T_ij.dtype)  # [n_batch, n_conf, n_atom*3]
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        T_supermatrix *= mask

        return T_supermatrix

    def compute_force(self, positions: Tensor, E_ind: Tensor) -> Tensor:

        assert positions.requires_grad
        pred_force = -torch.autograd.grad(E_ind, positions, grad_outputs=torch.ones_like(E_ind), create_graph=True)[0]
        return pred_force

    @torch.enable_grad()
    def forward(
        self,
        coords: Tensor,
        alpha,
        charge,
        count_node,
        count_12,
        count_13,
        count_14,
        count_15,
        node_12_idx,
        node_13_idx,
        node_14_idx,
        node_15_idx,
        ind14,
        ind15,
        pol_damping=None,
    ):
        """
        Solve the polarization equation with QR decomposition (increase numerical stability)

        Input include atomic positions, charge and polarizability

        :param positions: atomic positions with dim of [n_batch, n_atom, n_conf, 3]
        :param charge: atomic charges with dim of [n_batch, n_atom]
        :param polarizability: atomic polarizability with dim of [n_batch, n_atom]
        """
        device = coords.device

        # split batch
        node_batch = torch.arange(count_node.size()[0], dtype=torch.int64, device=device).repeat_interleave(count_node)
        alpha, _ = to_dense_batch(alpha, node_batch, fill_value=1.0)
        charge, _ = to_dense_batch(charge, node_batch)
        positions, node_mask = to_dense_batch(coords, node_batch, fill_rand=True,
                                              need_mask=True)  # [n_batch, n_atom, n_conf, 3], [n_batch, n_atom]

        batch_12, dense_index_12 = to_dense_index(node_12_idx, count_12, count_node)
        node_12_idx = (
            torch.concat((batch_12, batch_12)),
            torch.concat((dense_index_12[:, 0], dense_index_12[:, 1])),
            torch.concat((dense_index_12[:, 1], dense_index_12[:, 0])),
        )
        batch_13, dense_index_13 = to_dense_index(node_13_idx, count_13, count_node)
        node_13_idx = (
            torch.concat((batch_13, batch_13)),
            torch.concat((dense_index_13[:, 0], dense_index_13[:, 1])),
            torch.concat((dense_index_13[:, 1], dense_index_13[:, 0])),
        )
        batch_14, dense_index_14 = to_dense_index(node_14_idx, count_14, count_node)
        node_14_idx = (
            torch.concat((batch_14, batch_14)),
            torch.concat((dense_index_14[:, 0], dense_index_14[:, 1])),
            torch.concat((dense_index_14[:, 1], dense_index_14[:, 0])),
        )
        batch_15, dense_index_15 = to_dense_index(node_15_idx, count_15, count_node)
        node_15_idx = (
            torch.concat((batch_15, batch_15)),
            torch.concat((dense_index_15[:, 0], dense_index_15[:, 1])),
            torch.concat((dense_index_15[:, 1], dense_index_15[:, 0])),
        )

        n_batch, n_atom, n_conf, _ = positions.shape
        if pol_damping is None:
            dp = alpha.clone()
        else:
            # print("pol_damping", pol_damping.min(), pol_damping.max())
            dp, _ = to_dense_batch(pol_damping, node_batch, fill_value=1.0)
        dp = dp.unsqueeze(-1)
        pol_int = (dp * dp.transpose(-1, -2)).unsqueeze(1).expand(-1, n_conf, -1,
                                                                  -1)  # [n_batch, n_conf, n_atom, n_atom]
        scaler = coords.new_ones(
            (n_batch, n_atom, n_atom)) - torch.eye(n_atom, device=coords.device, dtype=coords.dtype).unsqueeze(0)
        scaler[node_12_idx] = 0.
        scaler[node_13_idx] = 0.
        scaler[node_14_idx] = ind14
        scaler[node_15_idx] = ind15
        smask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        scaler *= smask

        scaler = scaler.unsqueeze(1).expand(-1, n_conf, -1, -1)  # [n_batch, n_conf, n_atom, n_atom]

        if not positions.requires_grad:
            positions.requires_grad = True
        pos_i = positions.unsqueeze(1).expand(-1, n_atom, -1, -1, -1)
        pos_j = positions.unsqueeze(2).expand(-1, -1, n_atom, -1, -1)

        # Compute r_ij vectors and their norms
        r_ij = (pos_j - pos_i).movedim(-2, 1)  # [n_batch, n_conf, n_atom, n_atom, 3]
        r_norm = torch.norm(r_ij, dim=-1).clamp(min=1e-3)  # [n_batch, n_conf, n_atom, n_atom]

        D1, D2 = self.compute_damping_tensor(r_norm, pol_int)  # [n_batch, n_conf, n_atom, n_atom]

        T_u = self.compute_interaction_tensor(r_ij, r_norm, D1, D2, node_mask)  # [n_batch, n_conf, n_atom*3, n_atom*3]
        alpha_inv = self.compute_polarizability_tensor(n_batch, n_conf, alpha)  # [n_batch, n_conf, n_atom*3, n_atom*3]
        E_p = self.compute_electric_field(
            r_ij,
            r_norm,
            D1,
            scaler,
            charge,
        )  # [n_batch, n_conf, n_atom*3]
        A = alpha_inv + T_u  # [n_batch, n_conf, n_atom*3, n_atom*3]

        mu_p = torch.linalg.solve(A, E_p.unsqueeze(-1))  # pylint: disable=not-callable
        mu_p = mu_p.squeeze(-1)  # [n_batch, n_conf, n_atom*3]
        e_ind = -0.5 * (mu_p * E_p).sum(-1)  # [n_batch, n_conf]
        f_ind = self.compute_force(positions, e_ind)  # [n_batch, n_atom, n_conf, 3]
        f_ind = f_ind[node_mask]  # combine batch
        e_ind *= CHG_FACTOR
        f_ind *= CHG_FACTOR

        mu_p = mu_p.reshape(n_batch, n_conf, n_atom, 3).transpose(1, 2)  # [n_batch, n_atom, n_conf, 3]
        mu_p = mu_p[node_mask > 0].reshape(torch.sum(count_node), n_conf, 3)
        return E_p, mu_p, e_ind, f_ind, scaler, D1, D2


class PreExp6Pol(PreFFLayer):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            out_mlp_dims=(32, 3),  # (hidden, layers)
            act='gelu',
            c6_scale=1000.,
            combining_rule='LB',
            pol_damp_clip=1.0e-6,
            li_damp_clip=1.0e-6,
            fix_li_alpha=None,
            **configs):
        super().__init__(node_dim, edge_dim, **configs)

        self.c6_scale = c6_scale  # control grad value of c6_free
        self.pol_damp_clip = pol_damp_clip
        self.li_damp_clip = li_damp_clip
        self.v_free = Embedding.from_pretrained(torch.tensor(V_FREE).unsqueeze(-1), freeze=False)
        self.c6_free = Embedding.from_pretrained(torch.tensor(C6_FREE).unsqueeze(-1) / self.c6_scale, freeze=False)
        self.rvdw_free = Embedding.from_pretrained(torch.tensor(RVDW_FREE).unsqueeze(-1), freeze=False)
        self.alpha_free = Embedding.from_pretrained(torch.tensor(ALPHA_FREE).unsqueeze(-1), freeze=False)
        self.fix_li_alpha = fix_li_alpha
        if fix_li_alpha is not None:
            assert isinstance(fix_li_alpha, float)

        self.lamb_mlp = MLP(in_channels=node_dim,
                            hidden_channels=out_mlp_dims[0],
                            out_channels=1,
                            num_layers=out_mlp_dims[1],
                            norm=None,
                            act=act)
        self.eps_mlp = MLP(in_channels=node_dim,
                           hidden_channels=out_mlp_dims[0],
                           out_channels=1,
                           num_layers=out_mlp_dims[1],
                           norm=None,
                           act=act)
        self.ct_eps_mlp = MLP(in_channels=node_dim,
                              hidden_channels=out_mlp_dims[0],
                              out_channels=1,
                              num_layers=out_mlp_dims[1],
                              norm=None,
                              act=act)
        self.ct_lamb_mlp = MLP(in_channels=node_dim,
                               hidden_channels=out_mlp_dims[0],
                               out_channels=1,
                               num_layers=out_mlp_dims[1],
                               norm=None,
                               act=act)

        self.combining_rule = combining_rule

    def reset_parameters(self):
        self.lamb_mlp.reset_parameters()
        self.eps_mlp.reset_parameters()
        self.ct_lamb_mlp.reset_parameters()
        self.ct_eps_mlp.reset_parameters()

    def calc_ts(self, atomic_number, volume: Tensor):
        v_ratio = volume / self.v_free(atomic_number)
        c6 = self.c6_free(atomic_number) * v_ratio**2 * self.c6_scale
        rvdw = self.rvdw_free(atomic_number) * v_ratio**(4 / 21)
        alpha0 = self.alpha_free(atomic_number)
        alpha = alpha0 * v_ratio**(4 / 3)
        return c6, rvdw, alpha0, alpha

    def forward(self,
                data: MonoData,
                x_h: Tensor,
                e_h: Tensor,
                ff_parameters: dict[str, Tensor] = None) -> dict[str, Tensor]:

        volume = ff_parameters['PreChargeVolume.volumes']
        atomic_number = data.node_features[:, 0].long()
        c6, rvdw, alpha0, alpha = self.calc_ts(atomic_number, volume)

        alpha = torch.clamp(alpha, min=1e-6)

        if self.fix_li_alpha is not None:
            alpha = torch.where(atomic_number.unsqueeze(-1) == 10, self.fix_li_alpha, alpha)

        ff_parameters['PreExp6Pol.c6'] = c6.clip(min=1e-6)
        ff_parameters['PreExp6Pol.rvdw'] = rvdw
        ff_parameters['PreExp6Pol.alpha'] = alpha
        ff_parameters['PreExp6Pol.alpha0'] = alpha0
        ff_parameters['PreExp6Pol.eps'] = torch.exp(self.eps_mlp(x_h))
        ff_parameters['PreExp6Pol.ct_eps'] = torch.exp(self.ct_eps_mlp(x_h))
        ff_parameters['PreExp6Pol.ct_lamb'] = torch.exp(self.ct_lamb_mlp(x_h))

        lamb = torch.exp(self.lamb_mlp(x_h)) * 5.
        ff_parameters['PreExp6Pol.lambda'] = lamb
        damping = alpha.clone().squeeze(-1)
        # li damp clip
        if self.li_damp_clip > 0.:
            if not isinstance(self.li_damp_clip, float):
                li_pd = self.li_damp_clip.unsqueeze(-1).expand(damping.shape)
            else:
                li_pd = torch.clip(damping, min=self.li_damp_clip)
            damping = torch.where(data.node_features[:, 0] == 10, li_pd, damping)
        pol_damping = torch.clip(damping, min=self.pol_damp_clip)
        ff_parameters['PreExp6Pol.pol_damping'] = pol_damping.clone()
        return ff_parameters


class Exp6Pol(FFLayer):
    """ Modified from apple&p.
        Reference: 
        Borodin and Smith - 2006 - Development of Many-Body Polarizable Force Fields, DOI: 10.1021/jp055079e
        Borodin - 2009 - Polarizable Force Field Development and Molecular, DOI: 10.1021/jp905220k
    """

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 vdw14: float = 0.5,
                 charge14: float = 0.5,
                 ind14: float = 0.5,
                 ind15: float = 1.0,
                 pol_damping_factor=0.39,
                 disp_damping_factor=0.4,
                 s12=0.,
                 calc_pol=True,
                 combining_rule='LB',
                 **configs):
        super().__init__(node_dim, edge_dim)
        self.vdw14 = vdw14
        self.charge14 = charge14
        self.ind14 = ind14
        self.ind15 = ind15
        self.dipole_solver = PolarizationSolver(pol_damping_factor)
        self.disp_damping_factor = disp_damping_factor
        self.s12 = s12
        self.calc_pol = calc_pol
        self.nuclear_charge = Embedding.from_pretrained(torch.tensor(SUPPORTED_ELEMENTS,
                                                                     dtype=torch.float32).unsqueeze(-1),
                                                        freeze=True)

        assert combining_rule in ['LB', 'GM']
        self.combining_rule = combining_rule

    def reset_parameters(self):
        pass

    @staticmethod
    def calc_dist(coords: Tensor, node_idx: LongTensor) -> tuple[Tensor]:
        ai, aj = node_idx[:, 0], node_idx[:, 1]  # [npairs]
        r, r_vec = get_distance_vec(coords[ai], coords[aj])
        return ai, aj, r, r_vec

    @staticmethod
    def combining(ai, aj, lamb, c6, r0, combining_rule, eps=None) -> tuple[Tensor]:
        """ Lorentz-Berthelot (LB) and geometric (GM) combining rule """
        r0_i, r0_j = r0[ai], r0[aj]
        if combining_rule == 'LB':
            r0_ij = ((r0_i + r0_j) / 2).unsqueeze(-1)
        elif combining_rule == 'GM':
            r0_ij = torch.sqrt(r0_i * r0_j).unsqueeze(-1)
        else:
            raise NotImplementedError(f'combining_rule {combining_rule}')

        lamb_i, lamb_j = lamb[ai], lamb[aj]
        c6i, c6j = c6[ai], c6[aj]
        lamb_ij = torch.sqrt((lamb_i * lamb_j).sum(dim=-1)).unsqueeze(-1)
        eps_i, eps_j = eps[ai], eps[aj]
        eps_ij = torch.sqrt((eps_i * eps_j).sum(dim=-1)).unsqueeze(-1)
        c6ij = torch.sqrt(c6i * c6j).unsqueeze(-1)
        return lamb_ij, eps_ij, r0_ij, c6ij

    def calc_disp(self, coords, r, r_vec, ai, aj, counts, lamb, eps, r0, c6) -> tuple[Tensor]:
        """
        C6 with D3 damping
        U(r) = - eps * lamb / (c + (r/r0)^6) = - eps * lamb * r0^6 / (c * r0^6 + r^6)
        dU/dr = 6 * eps * lamb * r0^6 * r^5 / (c * r0^6 + r^6)^2
        """
        nconfs = coords.shape[1]

        r06 = r0**6
        invr6 = 1.0 / (self.disp_damping_factor * r06 + r**6)  # [npairs, nconfs]
        pair_energy = -c6 * invr6
        disp_energy = reduce_counts(pair_energy, counts)  # [batch_size, nconfs]
        disp_forces = torch.zeros_like(coords)
        pair_force = 6 * c6 * r**5 * invr6**2
        pair_force = (pair_force / r).unsqueeze(-1) * r_vec
        disp_forces.scatter_add_(0, ai.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
        disp_forces.scatter_add_(0, aj.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)

        return disp_energy, disp_forces

    def calc_rep(
        self,
        coords,
        r,
        r_vec,
        ai,
        aj,
        counts,
        lamb,
        eps,
        r0,
        c6=None,
    ) -> tuple[Tensor]:
        """
        exp repulsion
        U(r) = 6 * eps * exp(lamb * (1 - r / r0))
        dU/dr = -6 * eps * lamb / r0 * exp(lamb * (1 - r / r0))
        """
        r = torch.clip(r, min=1e-4)
        nconfs = coords.shape[1]
        pe0 = 6 * eps * torch.exp(lamb * (1 - r / r0))
        pair_energy = pe0.clone()
        if self.s12 > 0.:
            pe1 = (self.s12 / r)**12
            pair_energy += pe1

        rep_energy = reduce_counts(pair_energy, counts)  # [batch_size, nconfs]

        pair_force = -lamb / r0 * pe0
        if self.s12 > 0.:
            pair_force = pair_force - 12 * pe1 / r

        pair_force = (pair_force / r).unsqueeze(-1) * r_vec
        rep_forces = torch.zeros_like(coords)
        rep_forces.scatter_add_(0, ai.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
        rep_forces.scatter_add_(0, aj.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)
        return rep_energy, rep_forces

    def calc_induction(
        self,
        data: MonoData,
        charge,
        alpha,
        cluster=False,
        pol_damping=None,
    ) -> tuple[Tensor]:
        """
        Compute the induction interaction
        """
        E_p, mu_p, e_ind, f_ind, scaler, lambda3, lambda5 = self.dipole_solver(
            data.coords,
            alpha,
            charge,
            data.get_count("node", idx=None, cluster=cluster),
            data.get_count('nonbonded12', idx=None, cluster=cluster),
            data.get_count('nonbonded13', idx=None, cluster=cluster),
            data.get_count('nonbonded14', idx=None, cluster=cluster),
            data.get_count('nonbonded15', idx=None, cluster=cluster),
            data.inc_node_nonbonded12.long(),
            data.inc_node_nonbonded13.long(),
            data.inc_node_nonbonded14.long(),
            data.inc_node_nonbonded15.long(),
            self.ind14,
            self.ind15,
            pol_damping=pol_damping,
        )

        return E_p, mu_p, e_ind, f_ind, scaler, lambda3, lambda5

    def calc_perm(self, coords, ai, aj, counts, charge, r, r_vec) -> tuple[Tensor]:
        """
        Compute the interaction between permanent charges
        """

        nconfs = coords.shape[1]
        qi, qj = charge[ai], charge[aj]
        invr12 = 1.0 / r  # [npairs, nconfs]

        chg_energy = CHG_FACTOR * invr12 * (qi * qj).unsqueeze(-1)  # [npairs, nconfs]
        chg_energy = reduce_counts(chg_energy, counts)  # [batch_size, nconfs]

        chg_forces = torch.zeros_like(coords)
        # IMPORTANT: negative sign
        # [npairs, nconfs, 3]
        pair_force = -CHG_FACTOR * (torch.pow(invr12, 3) * (qi * qj).unsqueeze(-1)).unsqueeze(-1) * r_vec
        chg_forces.scatter_add_(0, ai.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
        chg_forces.scatter_add_(0, aj.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)

        return chg_energy, chg_forces

    def calc_cluster_elec_energy(self,
                                 data: ClusterData,
                                 charge: Tensor,
                                 dipole: Tensor,
                                 scaler: Tensor,
                                 lambda3: Tensor,
                                 lambda5: Tensor,
                                 exlude_charge=True,
                                 cluster=True):
        """ Calculate electrostatic energy of the cluster, up to dipole interaction.
            charge: [natom_all]
            dipole: [natom_all, n_conf, 3]
            scaler: [n_batch, n_conf, n_atom, n_atom]
            lambda3, lambda5: [n_batch, n_conf, n_atom, n_atom]
        """
        coords = data.coords
        device = coords.device

        # split batch
        count_node = data.get_count("node", idx=None, cluster=cluster)
        node_batch = torch.arange(count_node.size()[0], dtype=torch.int64, device=device).repeat_interleave(count_node)
        charges, _ = to_dense_batch(charge, node_batch)  # [n_batch, n_atom]
        positions, node_mask = to_dense_batch(coords, node_batch, fill_rand=True,
                                              need_mask=True)  # [n_batch, n_atom, n_conf, 3], [n_batch, n_atom]
        dipoles, _ = to_dense_batch(dipole, node_batch, fill_rand=True)  # [n_batch, n_atom, n_conf, 3]

        n_batch, n_atom, n_conf, _ = dipoles.shape
        charges = charges.view(n_batch, n_atom, 1, 1).repeat(1, 1, n_conf, 1)  # [n_batch, n_atom, n_conf, 1]
        M = torch.concat([charges, dipoles], dim=-1).movedim(-2, 1).reshape(n_batch, n_conf,
                                                                            -1)  # [n_batch, n_conf, n_atom * 4]

        pos_i = positions.unsqueeze(1).expand(-1, n_atom, -1, -1, -1)
        pos_j = positions.unsqueeze(2).expand(-1, -1, n_atom, -1, -1)
        r_ij = (pos_i - pos_j).movedim(-2, 1)  # [n_batch, n_conf, n_atom, n_atom, 3]
        r = (r_ij**2).sum(-1).sqrt().clamp(min=1e-6)  # [n_batch, n_conf, n_atom, n_atom]
        r2 = r.pow(2)
        r3 = r2 * r
        r5 = r3 * r2
        l3_r3 = lambda3 / r3
        T1 = -r_ij * l3_r3.unsqueeze(-1)  #  [n_batch, n_conf, n_atom, n_atom, 3]

        rij_outer = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
        eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(n_batch, n_conf, n_atom, n_atom, 3, 3)
        l5 = lambda5.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3, 3)
        T2 = -3 * rij_outer * l5 / r5.unsqueeze(-1).unsqueeze(-1) + eye * l3_r3.unsqueeze(-1).unsqueeze(-1)

        T = torch.zeros((n_batch, n_conf, n_atom, n_atom, 4, 4), dtype=r.dtype, device=r.device)
        if not exlude_charge:
            T[..., 0, 0] = 1 / r
        T[..., 0, 1:] = T1
        T[..., 1:, 0] = -T1
        T[..., 1:, 1:] = T2
        mask = node_mask.unsqueeze(1).expand(-1, n_conf, -1).to(T.dtype)  # [n_batch, n_conf, n_atom]
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [n_batch, n_conf, n_atom, n_atom]
        T = T * scaler.unsqueeze(-1).unsqueeze(-1) * mask.unsqueeze(-1).unsqueeze(-1)

        M = M.reshape(n_batch, n_conf, n_atom, 4).unsqueeze(-3).expand(n_batch, n_conf, n_atom, n_atom, 4)
        U = M.transpose(-2, -3).unsqueeze(-2) @ T @ M.unsqueeze(-1)
        U = CHG_FACTOR * U.squeeze(-1).squeeze(-1)
        U = 0.5 * U.sum(dim=-1).sum(dim=-1)
        return U

    def calc_charge_transfer(
        self,
        coords,
        ff_parameters,
        ai,
        aj,
        counts,
        r,
        r_vec,
        r0,
    ):
        """
        U = - eps * exp(- lambda * (r / r0))
        """
        r0i, r0j = r0[ai].unsqueeze(-1), r0[aj].unsqueeze(-1)
        eps = ff_parameters['PreExp6Pol.ct_eps']
        lamb = ff_parameters['PreExp6Pol.ct_lamb']
        eps_i, eps_j = eps[ai], eps[aj]
        lamb_i, lamb_j = lamb[ai], lamb[aj]
        eps_ij = (eps_i * eps_j).sqrt()
        lamb_ij = (lamb_i * lamb_j).sqrt()
        r0_ij = (r0i + r0j) / 2

        pair_energy = -eps_ij * r.pow(-4) * torch.exp(-(lamb_ij * r / r0_ij)**3)
        ct_energy = reduce_counts(pair_energy, counts)
        pair_force = pair_energy * (-3.0 * (lamb_ij * r / r0_ij)**3 / r - 4 / r)
        pair_force = (pair_force / r).unsqueeze(-1) * r_vec
        ct_forces = torch.zeros_like(coords)
        nconfs = coords.shape[1]
        ct_forces.scatter_add_(0, ai.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
        ct_forces.scatter_add_(0, aj.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)
        return ct_energy, ct_forces

    def forward(
        self,
        data: Union[MonoData, ClusterData],
        x_h: Tensor,
        e_h: Tensor,
        ff_parameters: dict[str, Tensor],
        cluster: bool = False,
    ):

        coords = data.coords
        confmask = data.confmask_cluster if cluster else data.confmask
        n_atom = data.get_count('node', idx=None, cluster=cluster)
        confmask_forces = confmask.repeat_interleave(n_atom, 0).unsqueeze(-1)
        surfix = '_cluster' if cluster else ''

        charge = ff_parameters['PreChargeVolume.charges'].squeeze(-1)
        lamb = ff_parameters['PreExp6Pol.lambda']
        c6 = ff_parameters['PreExp6Pol.c6'].squeeze(-1)
        r0 = ff_parameters['PreExp6Pol.rvdw'].squeeze(-1)
        alpha = ff_parameters['PreExp6Pol.alpha'].squeeze(-1).clone()
        eps = ff_parameters['PreExp6Pol.eps']
        e_static_ind, e_static_ind_split = None, None

        node14_idx = data.inc_node_nonbonded14.long()
        counts14 = data.get_count('nonbonded14', idx=None, cluster=cluster)

        nodeall_idx = data.inc_node_nonbonded_all_cluster.long() if cluster else data.inc_node_nonbonded_all.long()
        countsall = data.get_count('nonbonded_all', idx=None, cluster=cluster)

        # nonbonded 14
        ai, aj, r, r_vec = self.calc_dist(coords, node14_idx)
        if self.combining_rule in ['LB', 'GM']:
            params = self.combining(ai, aj, lamb, c6, r0, self.combining_rule, eps=eps)
        else:
            raise NotImplementedError(f'combining_rule {self.combining_rule}')

        rep_energy_14, rep_forces_14 = self.calc_rep(coords, r, r_vec, ai, aj, counts14, *params)
        disp_energy_14, disp_forces_14 = self.calc_disp(coords, r, r_vec, ai, aj, counts14, *params)
        chg_e_14, chg_f_14 = self.calc_perm(coords, ai, aj, counts14, charge, r, r_vec)

        # nonbonded all
        ai, aj, r, r_vec = self.calc_dist(coords, nodeall_idx)
        if self.combining_rule in ['LB', 'GM']:
            params = self.combining(ai, aj, lamb, c6, r0, self.combining_rule, eps=eps)
        else:
            raise NotImplementedError(f'combining_rule {self.combining_rule}')

        rep_energy_all, rep_forces_all = self.calc_rep(coords, r, r_vec, ai, aj, countsall, *params)
        disp_energy_all, disp_forces_all = self.calc_disp(coords, r, r_vec, ai, aj, countsall, *params)
        chg_e_all, chg_f_all = self.calc_perm(coords, ai, aj, countsall, charge, r, r_vec)

        ai15, aj15, r15, r_vec15 = self.calc_dist(coords, data.inc_node_nonbonded15.long())
        count15 = data.get_count('nonbonded15', idx=None, cluster=cluster)
        ct_energy, ct_force = self.calc_charge_transfer(
            coords,
            ff_parameters,
            ai,
            aj,
            countsall,
            r,
            r_vec,
            r0,
        )
        ct_energy15, ct_force15 = self.calc_charge_transfer(
            coords,
            ff_parameters,
            ai15,
            aj15,
            count15,
            r15,
            r_vec15,
            r0,
        )
        ct_energy -= ct_energy15
        ct_force -= ct_force15

        # nonbonded induction
        if self.calc_pol:
            pol_damping = ff_parameters['PreExp6Pol.pol_damping'].squeeze(-1)
            _, mu_p, induction_e, induction_f, scaler, lambda3, lambda5 = self.calc_induction(
                data,
                charge,
                alpha,
                cluster=cluster,
                pol_damping=pol_damping,
            )
            mu_p = mu_p * confmask_forces
            if not cluster:
                ff_parameters['Exp6Pol.mu_p'] = mu_p.clone()
                ff_parameters['Exp6Pol.ind_scaler'] = scaler.clone()
                ff_parameters['Exp6Pol.ind_lambda3'] = lambda3.clone()
                ff_parameters['Exp6Pol.ind_lambda5'] = lambda5.clone()

            if cluster:
                ff_parameters['Exp6Pol.mu_p_cluster'] = mu_p.clone()
                e_static_ind_split = self.calc_cluster_elec_energy(data,
                                                                   charge,
                                                                   ff_parameters['Exp6Pol.mu_p'],
                                                                   ff_parameters['Exp6Pol.ind_scaler'],
                                                                   ff_parameters['Exp6Pol.ind_lambda3'],
                                                                   ff_parameters['Exp6Pol.ind_lambda3'],
                                                                   exlude_charge=True,
                                                                   cluster=False)

                e_static_ind = self.calc_cluster_elec_energy(data,
                                                             charge,
                                                             ff_parameters['Exp6Pol.mu_p'],
                                                             scaler,
                                                             lambda3,
                                                             lambda5,
                                                             exlude_charge=True)
        else:
            induction_e, induction_f = 0., 0.

        ff_parameters['Exp6Pol.perm_chg_energy' + surfix] = (chg_e_all + self.charge14 * chg_e_14) * confmask
        ff_parameters['Exp6Pol.induction_energy' + surfix] = induction_e * confmask
        ff_parameters['Exp6Pol.rep_energy' + surfix] = (rep_energy_14 * self.vdw14 + rep_energy_all) * confmask
        ff_parameters['Exp6Pol.disp_energy' + surfix] = (disp_energy_14 * self.vdw14 + disp_energy_all) * confmask
        ff_parameters['Exp6Pol.perm_chg_forces' + surfix] = (chg_f_all + self.charge14 * chg_f_14) * confmask_forces
        ff_parameters['Exp6Pol.induction_forces' + surfix] = induction_f * confmask_forces
        ff_parameters['Exp6Pol.rep_forces' + surfix] = (rep_forces_14 * self.vdw14 + rep_forces_all) * confmask_forces
        ff_parameters['Exp6Pol.disp_forces' +
                      surfix] = (disp_forces_14 * self.vdw14 + disp_forces_all) * confmask_forces

        ff_parameters['Exp6Pol.energy' + surfix] = (ff_parameters['Exp6Pol.perm_chg_energy' + surfix] +
                                                    ff_parameters['Exp6Pol.induction_energy' + surfix] +
                                                    ff_parameters['Exp6Pol.rep_energy' + surfix] +
                                                    ff_parameters['Exp6Pol.disp_energy' + surfix])
        ff_parameters['Exp6Pol.forces' + surfix] = (ff_parameters['Exp6Pol.perm_chg_forces' + surfix] +
                                                    ff_parameters['Exp6Pol.induction_forces' + surfix] +
                                                    ff_parameters['Exp6Pol.rep_forces' + surfix] +
                                                    ff_parameters['Exp6Pol.disp_forces' + surfix])

        ff_parameters['Exp6Pol.ct_energy' + surfix] = ct_energy * confmask
        ff_parameters['Exp6Pol.ct_forces' + surfix] = ct_force * confmask_forces
        ff_parameters['Exp6Pol.energy' + surfix] += ff_parameters['Exp6Pol.ct_energy' + surfix]
        ff_parameters['Exp6Pol.forces' + surfix] += ff_parameters['Exp6Pol.ct_forces' + surfix]

        if cluster and 'Exp6Pol.disp_energy' in ff_parameters:
            nmols = data.get_count('mol', idx=None, cluster=True)
            batches = torch.arange(nmols.shape[0], device=nmols.device).repeat_interleave(nmols).unsqueeze(-1).expand(
                -1, chg_e_all.shape[1])
            template = torch.zeros_like(chg_e_all)

            ff_parameters['DISP'] = ff_parameters['Exp6Pol.disp_energy_cluster'] - template.clone().scatter_add_(
                0, batches, ff_parameters['Exp6Pol.disp_energy'])
            ff_parameters['PAULI'] = ff_parameters['Exp6Pol.rep_energy_cluster'] - template.clone().scatter_add_(
                0, batches, ff_parameters['Exp6Pol.rep_energy'])
            ff_parameters['ELEC'] = ff_parameters['Exp6Pol.perm_chg_energy_cluster'] - template.clone().scatter_add_(
                0, batches, ff_parameters['Exp6Pol.perm_chg_energy'])
            ind_0 = template.clone().scatter_add_(0, batches, ff_parameters['Exp6Pol.induction_energy'])
            ff_parameters['POLARIZATION'] = ff_parameters['Exp6Pol.induction_energy_cluster'] - ind_0

            e_dip = e_static_ind - template.clone().scatter_add_(0, batches, e_static_ind_split)
            ff_parameters['ELEC'] += e_dip
            ff_parameters['POLARIZATION'] -= e_dip

            ff_parameters['CHARGE_TRANSFER'] = ff_parameters['Exp6Pol.ct_energy_cluster'] - template.clone(
            ).scatter_add_(0, batches, ff_parameters['Exp6Pol.ct_energy'])

        return ff_parameters['Exp6Pol.energy' + surfix].clone(), ff_parameters['Exp6Pol.forces' + surfix].clone()
