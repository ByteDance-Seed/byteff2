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
import torch.nn as nn
from torch import LongTensor, Tensor
from torch_geometric.utils import cumsum, scatter

# pylint: disable=not-callable


def cosine_cutoff(values: torch.Tensor, lower: float, upper: float):
    cutoffs = 0.5 * (torch.cos((values - lower) / (upper - lower) * torch.pi) + 1.0)
    # remove contributions below the cutoff radius
    cutoffs = cutoffs * (values < upper).float()
    cutoffs = torch.where(values < lower, 1., cutoffs)
    return cutoffs


class CosineCutoff(nn.Module):

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        cutoffs = cosine_cutoff(distances, self.cutoff_lower, self.cutoff_upper)
        return cutoffs


class ExpNormalSmearing(nn.Module):

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=32, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value))**-2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha *
                                                                         (-dist + self.cutoff_lower)) - self.means)**2)


def reduce_counts(src: Tensor, counts: LongTensor, reduce: str = 'sum'):
    """
    Reduce values to batch accoring to counts.
    
    Example:
    src: [1, 2, 1, 1], counts: [3, 1] -->
    ret: [4, 1]
    """

    batch = torch.arange(len(counts), dtype=torch.int64, device=src.device)
    batch = torch.repeat_interleave(batch, counts, dim=0)
    ret = scatter(src, batch, dim=0, dim_size=len(counts), reduce=reduce)
    return ret


def batch_to_atoms(src: Tensor, batch: LongTensor) -> Tensor:
    """[bs, nconfs] -> [natoms, nconfs, 3]"""
    ret = torch.gather(src, 0, batch)
    ret = ret.unsqueeze(-1).expand(-1, -1, 3)
    return ret


def get_batch_idx(idx: int, counts: Tensor) -> int:
    cc = torch.cumsum(counts, dim=0)
    batch_idx = torch.where(idx < cc)[0][0]
    return batch_idx


def set_grad_max(grad: Tensor, max_step: float):
    if grad.abs().max() > max_step:
        grad *= max_step / grad.abs().max()
    return grad


def get_distance_vec(r1: Tensor, r2: Tensor) -> tuple[Tensor, Tensor]:
    r12vec = r2 - r1
    r12 = torch.linalg.vector_norm(r12vec, dim=-1)
    return r12, r12vec


def get_angle_vec(r0: Tensor, r1: Tensor, r2: Tensor, with_vec: bool = True):
    v0 = r1 - r0  # [nangles, n_conf, 3]
    v1 = r1 - r2  # [nangles, n_conf, 3]
    cross = torch.linalg.cross(v0, v1, dim=-1)  # [nangles, n_conf, 3]
    dot_v0_v1 = torch.linalg.vecdot(v0, v1)
    # atan2 generates nan when computing hessian for exact linear angle.
    angle = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), dot_v0_v1)

    if with_vec:
        # rp becomes singular when cross=0 (v0//v1), then f1/f3 become nan
        r0 = torch.linalg.vecdot(v0, v0)
        r1 = torch.linalg.vecdot(v1, v1)
        rp = torch.linalg.vector_norm(cross, dim=-1)  # [nangle, n_conf, 1]
        f1 = -torch.linalg.cross(v0, cross, dim=-1) / (r0 * rp).unsqueeze(-1)
        f3 = -torch.linalg.cross(cross, v1, dim=-1) / (r1 * rp).unsqueeze(-1)
    else:
        f1 = f3 = torch.zeros_like(r0)

    return angle, f1, f3


def get_dihedral_angle_vec(r0: Tensor, r1: Tensor, r2: Tensor, r3: Tensor, with_vec: bool = True):

    # use the method in gromacs 1234 <-> ijkl
    r_ij = r1 - r0
    r_kj = r1 - r2
    r_kl = r3 - r2
    m = torch.linalg.cross(r_ij, r_kj, dim=-1)
    n = torch.linalg.cross(r_kj, r_kl, dim=-1)
    w = torch.linalg.cross(m, n, dim=-1)
    wlen = torch.linalg.vector_norm(w, dim=-1)  # [ndihedrals, n_conf]
    s = torch.linalg.vecdot(m, n)
    phi = torch.atan2(wlen, s)  # [ndihedrals, n_conf]
    ipr = torch.linalg.vecdot(r_ij, n)  # [ndihedrals, n_conf]
    ipr = torch.where(torch.abs(ipr) > torch.finfo().eps, ipr, 1.0)
    phi = -phi * torch.sign(ipr)  # right hand sign

    if with_vec:
        iprm = torch.linalg.vecdot(m, m)
        iprn = torch.linalg.vecdot(n, n)
        nrkj2 = torch.linalg.vecdot(r_kj, r_kj)

        nrkj_1 = torch.rsqrt(nrkj2)  # [ndihedrals, n_conf]
        nrkj_2 = torch.square(nrkj_1)  # [ndihedrals, n_conf]
        nrkj = nrkj2 * nrkj_1  # [ndihedrals, n_conf]
        a = -nrkj / iprm  # [ndihedrals, n_conf]
        f_i = -a.unsqueeze(-1) * m  # [ndihedrals, n_conf, 3]
        b = nrkj / iprn  # [ndihedrals, n_conf]
        f_l = -b.unsqueeze(-1) * n  # [ndihedrals, n_conf, 3]
        p = torch.linalg.vecdot(r_ij, r_kj)  # [ndihedrals, n_conf]
        p *= nrkj_2  # [ndihedrals, n_conf]
        q = torch.linalg.vecdot(r_kl, r_kj)  # [ndihedrals, n_conf]
        q *= nrkj_2  # [ndihedrals, n_conf]

        uvec = p.unsqueeze(-1) * f_i  # [ndihedrals, n_conf, 3]
        vvec = q.unsqueeze(-1) * f_l  # [ndihedrals, n_conf, 3]
        svec = uvec - vvec  # [ndihedrals, n_conf, 3]
        f_j = (f_i - svec)  # [ndihedrals, n_conf, 3]
        f_k = (f_l + svec)  # [ndihedrals, n_conf, 3]
    else:
        f_i = f_j = f_k = f_l = torch.zeros_like(r0)

    return phi, f_i, -f_j, -f_k, f_l


def to_dense_batch(x: Tensor, batch: Tensor, fill_value=0., fill_rand=False, need_mask=False):
    """
    modified from https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/utils/_to_dense_batch.py

    x: [n_atom, ...]
    batch: [n_atom]
    fill_rand: fill with random values
    """

    batch_size = int(batch.max()) + 1
    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size, reduce='sum')  # [n_batch]
    cum_nodes = cumsum(num_nodes)

    max_num_nodes = int(num_nodes.max())
    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]

    if fill_rand:
        out = torch.rand(size, device=x.device, dtype=x.dtype)
    else:
        out = torch.as_tensor(fill_value, device=x.device, dtype=x.dtype)
        out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    if need_mask:
        mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
        mask[idx] = 1
        mask = mask.view(batch_size, max_num_nodes)
    else:
        mask = None

    return out, mask


def to_dense_index(index: Tensor, count: Tensor, inc: Tensor):
    batch = torch.arange(count.size()[0], device=count.device).repeat_interleave(count)
    inc_cumsum = cumsum(inc)[:-1]
    inc_cumsum = inc_cumsum.repeat_interleave(count)
    raw_index = index - inc_cumsum.unsqueeze(-1)
    return batch, raw_index
