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
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

from byteff2.data.data import Data, MonoData
from byteff2.utils.definitions import fudgeLJ, fudgeQQ, NBMMParam

from .base import FFLayer, PreFFLayer
from .ff_kernels import ClassicalForceField as CFF


class PreLJEs(PreFFLayer):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        pre_mlp_dims=(32, 32, 3),  # (hidden, out, layers)
        out_mlp_dims=(32, 3),  # (hidden, layers)
        act="gelu",
        tanh_output=4.0,
        **configs,
    ):
        super().__init__(node_dim, edge_dim)

        self.tanh_output = tanh_output
        self.sigma_mlp = MLP(
            in_channels=node_dim,
            hidden_channels=out_mlp_dims[0],
            out_channels=1,
            num_layers=out_mlp_dims[1],
            norm=None,
            act=act,
        )
        self.epsilon_mlp = MLP(
            in_channels=node_dim,
            hidden_channels=out_mlp_dims[0],
            out_channels=1,
            num_layers=out_mlp_dims[1],
            norm=None,
            act=act,
        )
        self.charge_pre_mlp = MLP(
            in_channels=node_dim * 2 + edge_dim,
            hidden_channels=pre_mlp_dims[0],
            out_channels=pre_mlp_dims[1],
            num_layers=pre_mlp_dims[2],
            norm=None,
            act=act,
        )
        self.charge_out_mlp = MLP(
            in_channels=pre_mlp_dims[1],
            hidden_channels=out_mlp_dims[0],
            out_channels=1,
            num_layers=out_mlp_dims[1],
            bias=False,
            norm=None,
            act="tanh",
        )

    def reset_parameters(self):
        self.sigma_mlp.reset_parameters()
        self.epsilon_mlp.reset_parameters()
        self.charge_pre_mlp.reset_parameters()
        self.charge_out_mlp.reset_parameters()

    def _bcc_charge(self, graph: MonoData, x_h: torch.Tensor, e_h: torch.Tensor):
        node_idx = graph["inc_node_bond"].long()
        edge_idx = graph["inc_edge_bond"].long()
        xs = [x_h[node_idx[:, 0]], e_h[edge_idx[:, 0]], x_h[node_idx[:, 1]]]
        xs = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))
        y0 = self.charge_pre_mlp(xs[0])
        y1 = self.charge_pre_mlp(xs[1])
        bcc = self.charge_out_mlp(y0 - y1).squeeze(-1)
        bcc = torch.tanh(bcc) * self.tanh_output

        # formal charge
        charge = graph.node_features[:, 2].clone().to(bcc.dtype)
        # average symmetric atoms
        equiv_idx = graph.inc_node_equiv.long()
        charge = scatter(charge, equiv_idx, 0, reduce="mean")[equiv_idx]
        # add bcc
        charge.scatter_add_(0, node_idx[:, 0], bcc)
        charge.scatter_add_(0, node_idx[:, 1], -bcc)
        return charge.unsqueeze(-1)

    def forward(
        self, data: MonoData, x_h: torch.Tensor, e_h: torch.Tensor, ff_parameters: dict[str, torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        ff_parameters = {}
        ff_parameters["PreLJEs.charges"] = self._bcc_charge(data, x_h, e_h)
        ff_parameters["PreLJEs.sigma"] = self.tanh_output * torch.tanh(self.sigma_mlp(x_h)) + self.tanh_output
        ff_parameters["PreLJEs.epsilon"] = self.tanh_output * torch.tanh(self.epsilon_mlp(x_h)) + self.tanh_output
        return ff_parameters


class LJEs(FFLayer):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        rep_power: int = 12,
        charge14: float = fudgeQQ,
        lj14: float = fudgeLJ,
        **configs,
    ):
        super().__init__(node_dim, edge_dim)
        assert rep_power in [12]
        self.charge14 = charge14
        self.lj14 = lj14

    def reset_parameters(self):
        pass

    def forward(
        self,
        data: Data,
        x_h: torch.Tensor,
        e_h: torch.Tensor,
        ff_parameters: dict[str, torch.Tensor],
        cluster: bool = False,
    ):

        coords = data.coords
        nonbonded_params = {
            NBMMParam.sigma: ff_parameters["PreLJEs.sigma"].squeeze(-1),
            NBMMParam.epsilon: ff_parameters["PreLJEs.epsilon"].squeeze(-1),
            NBMMParam.charges: ff_parameters["PreLJEs.charges"].squeeze(-1),
        }

        # nonbonded 14
        node_idx = data.inc_node_nonbonded14.long()
        counts = data.get_count("nonbonded14", idx=None, cluster=cluster)
        nonbonded14_energy, nonbonded14_forces, _ = CFF.calc_nonbonded14(
            coords,
            nonbonded_params,
            node_idx,
            counts,
            charge14_scale=self.charge14,
            lj14_scale=self.lj14,
        )

        # nonbonded all
        node_idx = data.inc_node_nonbonded_all_cluster.long() if cluster else data.inc_node_nonbonded_all.long()
        counts = data.get_count("nonbonded_all", idx=None, cluster=cluster)
        nonbonded_all_energy, nonbonded_all_forces, _ = CFF.calc_nonbonded_all(
            coords,
            nonbonded_params,
            node_idx,
            counts,
        )

        energy = nonbonded_all_energy + nonbonded14_energy
        forces = nonbonded_all_forces + nonbonded14_forces

        confmask = data.confmask_cluster if cluster else data.confmask
        n_atom = data.get_count("node", idx=None, cluster=cluster)
        confmask_forces = confmask.repeat_interleave(n_atom, 0).unsqueeze(-1)
        energy *= confmask
        forces *= confmask_forces

        return energy, forces
