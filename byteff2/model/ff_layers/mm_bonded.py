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
from torch import nn, Tensor
from torch_geometric.nn import MLP

from byteff2.data.data import ClusterData, MonoData
from byteff2.utils.definitions import MMParam, PROPERTORSION_TERMS

from .base import FFLayer, PreFFLayer
from .ff_kernels import ClassicalForceField as CFF
from .utils import set_grad_max


class CustomClamp(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, _input, _min=None, _max=None):
        return _input.clamp(min=_min, max=_max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


custom_clamp = CustomClamp.apply

term_shapes = {"bond": 2, "angle": 3, "proper": 4, "improper": 2}


class PreMMBonded(PreFFLayer):
    term_param_map = {
        "bond": {"bond_k": 1, "bond_r0": 1},
        "angle": {"angle_k": 1, "angle_d0": 1},
        "proper": {
            "proper_k": PROPERTORSION_TERMS,
            # 'proper_d0': PROPERTORSION_TERMS
        },
        "improper": {
            "improper_k": 1,
            # 'improper_d0': 1  # fix to pi
        },
    }

    param_std_mean_range = {
        "bond_k": (200.0, 700.0, 80.0, 4000.0),
        "bond_r0": (0.19, 1.3, 0.5, 5.0),
        "angle_k": (60.0, 130.0, 40.0, 1000.0),
        "angle_d0": (10.0, 120.0, 20.0, 180.0),
        "proper_k": (1.2, 0.4, -20.0, 20.0),
        "improper_k": (3.7, 4.3, 0.0, 20.0),
    }

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        pre_mlp_dims=(32, 32, 3),  # (hidden, out, layers)
        post_mlp_dims=(32, 32, 3),  # (hidden, out, layers)
        out_mlp_dims=(32, 3),  # (hidden, layers)
        act="gelu",
        tanh_output=15.0,
        grad_max=None,
        **configs,
    ):
        super().__init__(node_dim, edge_dim)

        self.tanh_output = tanh_output
        self.grad_max = grad_max
        self.pre_mlp: dict[str, MLP] = nn.ModuleDict()
        self.post_mlp: dict[str, MLP] = nn.ModuleDict()
        self.out_mlp: dict[str, MLP] = nn.ModuleDict()

        for term, shape in term_shapes.items():
            self.pre_mlp[term] = MLP(
                in_channels=shape * node_dim + (shape - 1) * edge_dim,
                hidden_channels=pre_mlp_dims[0],
                out_channels=pre_mlp_dims[1],
                num_layers=pre_mlp_dims[2],
                norm=None,
                act=act,
            )
            self.post_mlp[term] = MLP(
                in_channels=pre_mlp_dims[1],
                hidden_channels=pre_mlp_dims[0],
                out_channels=pre_mlp_dims[1],
                num_layers=pre_mlp_dims[2],
                norm=None,
                act=act,
            )
        for term, params in self.term_param_map.items():
            for p, w in params.items():
                self.out_mlp[p] = MLP(
                    in_channels=post_mlp_dims[1],
                    hidden_channels=out_mlp_dims[0],
                    out_channels=w,
                    num_layers=out_mlp_dims[1],
                    norm=None,
                    act=act,
                )

    def reset_parameters(self):
        for module in self.pre_mlp.values():
            module.reset_parameters()
        for module in self.post_mlp.values():
            module.reset_parameters()
        for module in self.out_mlp.values():
            module.reset_parameters()

    def _symmetric_pooling(self, x_h: Tensor, e_h: Tensor, graph: MonoData) -> dict[str, Tensor]:
        """post_mlp -> symmetry-preserving pooling with bond -> post_mlp -> out_mlp"""
        ret = {}
        for term in term_shapes:
            node_idx = graph[f"inc_node_{term}"].long()
            edge_idx = graph[f"inc_edge_{term}"].long()
            width = node_idx.shape[1]

            xs = []
            for i in range(width):
                xs.append(x_h[node_idx[:, i]])
                if i < width - 1:
                    xs.append(e_h[edge_idx[:, i]])

            if term != "improper":
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
            for k, value in params.items():
                p = k.split(".")[1]
                param_value = value
                if p in self.grad_max:
                    param_value = value.clone()
                    param_value.register_hook(partial(set_grad_max, max_step=self.grad_max[p]))
                param_value.retain_grad()
                new_params[k] = param_value
        else:
            new_params = params
        return new_params

    def patch_bonded_terms(self, data: MonoData, param: Tensor, term: str):
        """Mask bonded MM parameters at hypervalent atoms.

        Model forward replaces the affected bond / angle / proper / improper
        parameters with constants (bond/angle force constants get a fixed
        positive value matching the write-time hypervalent patch;
        equilibrium values and proper/improper amplitudes are zeroed). The
        substitution is implemented with ``torch.where`` so gradients on the
        masked positions are cut. Applied after ``custom_clamp`` so the
        substitution cannot be undone by clamping.

        Mask sources (constructed in byteff2/data/data.py):
            patch_bond_mask: marks bonds touching any hypervalent atom (P/S
                with degree >= 5).
            patch_angle_mask: marks angles centered on any hypervalent atom.
            patch_proper_mask, patch_improper_mask: marks proper/improper
                torsions involving any hypervalent atom, including ring cases.
        """
        if (
            "patch_bond_mask" not in data
            and "patch_angle_mask" not in data
            and "patch_proper_mask" not in data
            and "patch_improper_mask" not in data
        ):
            return param

        # Constants must match the write-time hypervalent bonded patch.
        bk, ak = 1000.0, 278.44
        if term.startswith("bond_") and "patch_bond_mask" in data:
            mask = data.patch_bond_mask.unsqueeze(-1) > 0.9
            if term in ("bond_k", "bond_k1"):
                fill = torch.full_like(param, bk)
            else:
                # bond_r0, bond_k2: zero out (k_sum still bk > 0 in conj form).
                fill = torch.zeros_like(param)
            d = torch.where(mask, fill, param)
        elif term.startswith("angle_") and "patch_angle_mask" in data:
            mask = data.patch_angle_mask.unsqueeze(-1) > 0.9
            if term in ("angle_k", "angle_k1"):
                fill = torch.full_like(param, ak)
            else:
                fill = torch.zeros_like(param)
            d = torch.where(mask, fill, param)
        elif term == "proper_k" and "patch_proper_mask" in data:
            d = torch.where(data.patch_proper_mask.unsqueeze(-1) > 0.9, torch.zeros_like(param), param)
        elif term == "improper_k" and "patch_improper_mask" in data:
            d = torch.where(data.patch_improper_mask.unsqueeze(-1) > 0.9, torch.zeros_like(param), param)
        else:
            d = param
        return d

    def forward(
        self,
        data: MonoData,
        x_h: Tensor,
        e_h: Tensor,
        ff_parameters: dict[str, Tensor] = None,
        do_patch: bool = False,
    ) -> dict[str, Tensor]:
        params = self._symmetric_pooling(x_h, e_h, data)
        ff_parameters = {}
        for term in params:
            d = params[term]
            if self.tanh_output > 0.0:
                d = self.tanh_output * torch.tanh(d)
            const = self.param_std_mean_range[term]
            d = d * const[0] + const[1]
            d = custom_clamp(d, const[2], const[3])
            d = self.patch_bonded_terms(data, d, term)
            ff_parameters[f"{type(self).__name__}.{term}"] = d

        if "proper_mask" in data:
            ff_parameters[f"{type(self).__name__}.proper_k"] *= data.proper_mask.unsqueeze(-1)

        ff_parameters = self.set_grad_max(ff_parameters)
        return ff_parameters


class MMBonded(FFLayer):
    def reset_parameters(self):
        pass

    def calc_bond(
        self,
        graph: MonoData,
        ff_params: dict,
        cluster=False,
        calc_partial_hessian=False,
    ):
        k = ff_params[f"Pre{type(self).__name__}.bond_k"]
        r0 = ff_params[f"Pre{type(self).__name__}.bond_r0"]
        coords = graph.coords
        bond_params = {MMParam.bond_k: k, MMParam.bond_r0: r0}
        node_idx = graph.inc_node_bond.long()
        counts = graph.get_count("bond", idx=None, cluster=cluster)
        return CFF.calc_bond(coords, bond_params, node_idx, counts, calc_partial_hessian)

    def calc_angle(
        self,
        graph: MonoData,
        ff_params: dict,
        cluster=False,
        calc_partial_hessian=False,
    ):
        k = ff_params[f"Pre{type(self).__name__}.angle_k"]
        d0 = ff_params[f"Pre{type(self).__name__}.angle_d0"]
        angle_params = {MMParam.angle_k: k, MMParam.angle_d0: d0}
        coords = graph.coords
        node_idx = graph.inc_node_angle.long()
        counts = graph.get_count("angle", idx=None, cluster=cluster)
        return CFF.calc_angle(coords, angle_params, node_idx, counts, calc_partial_hessian)

    def calc_proper(
        self,
        graph: MonoData,
        ff_params: dict,
        cluster=False,
        calc_partial_hessian=False,
    ):
        k = ff_params[f"Pre{type(self).__name__}.proper_k"]
        proper_params = {MMParam.proper_k: k}
        node_idx = graph.inc_node_proper.long()
        counts = graph.get_count("proper", idx=None, cluster=cluster)
        return CFF.calc_proper(graph.coords, proper_params, node_idx, counts, calc_partial_hessian)

    def calc_improper(
        self,
        graph: MonoData,
        ff_params: dict,
        cluster=False,
        calc_partial_hessian=False,
    ):
        k = ff_params[f"Pre{type(self).__name__}.improper_k"]
        improper_params = {MMParam.improper_k: k}
        node_idx = graph.inc_node_improper.long()
        counts = graph.get_count("improper", idx=None, cluster=cluster)
        return CFF.calc_improper(graph.coords, improper_params, node_idx, counts, calc_partial_hessian)

    def forward(
        self,
        data: Union[MonoData, ClusterData],
        x_h: Tensor,
        e_h: Tensor,
        ff_parameters: dict[str, Tensor],
        cluster: bool = False,
        calc_partial_hessian=False,
    ):

        energy, forces = 0.0, 0.0
        if calc_partial_hessian:
            data.coords.requires_grad = True

        for term in term_shapes:
            e_term, f_term, h_term = getattr(self, f"calc_{term}")(
                data,
                ff_parameters,
                cluster=cluster,
                calc_partial_hessian=calc_partial_hessian,
            )
            energy += e_term
            forces += f_term
            ff_parameters[f"{type(self).__name__}.{term}_energy"] = e_term
            ff_parameters[f"{type(self).__name__}.{term}_forces"] = f_term
            ff_parameters[f"{type(self).__name__}.{term}_hessian"] = h_term

        confmask = data.confmask_cluster if cluster else data.confmask
        n_atom = data.get_count("node", idx=None, cluster=cluster)
        confmask_forces = confmask.repeat_interleave(n_atom, 0).unsqueeze(-1).repeat(1, 1, 3)
        energy *= confmask
        forces *= confmask_forces

        return energy, forces


class PreMMBondedConj(PreMMBonded):
    term_param_map = {
        "bond": {"bond_k1": 1, "bond_k2": 1},
        "angle": {"angle_k1": 1, "angle_k2": 1},
        "proper": {
            "proper_k": PROPERTORSION_TERMS,
            # 'proper_d0': PROPERTORSION_TERMS
        },
        "improper": {
            "improper_k": 1,
            # 'improper_d0': 1  # fix to pi
        },
    }
    param_std_mean_range = {
        "bond_k1": (50.0, 1000.0, 0.0, 4000.0),
        "bond_k2": (25.0, 500.0, 0.0, 4000.0),
        "angle_k1": (5.0, 100.0, 0.0, 400.0),
        "angle_k2": (5.0, 100.0, 0.0, 400.0),
        "proper_k": (1.2, 0.4, -20.0, 20.0),
        "improper_k": (3.7, 4.3, 0.0, 40.0),
    }

    bond_b1 = 0.5
    bond_b2 = 4.0
    angle_b1 = 0.25 * torch.pi
    angle_b2 = 1.05 * torch.pi


class MMBondedConj(MMBonded):
    def calc_bond(
        self,
        graph: MonoData,
        ff_params: dict,
        cluster=False,
        calc_partial_hessian=False,
    ):
        k1 = ff_params[f"Pre{type(self).__name__}.bond_k1"]
        k2 = ff_params[f"Pre{type(self).__name__}.bond_k2"]
        b1, b2 = PreMMBondedConj.bond_b1, PreMMBondedConj.bond_b2
        coords = graph.coords
        node_idx = graph.inc_node_bond.long()
        counts = graph.get_count("bond", idx=None, cluster=cluster)
        return CFF.calc_conj_bond(coords, k1, k2, b1, b2, node_idx, counts, calc_partial_hessian)

    def calc_angle(self, graph: MonoData, ff_params: dict, cluster=False, calc_partial_hessian=False):
        k1 = ff_params[f"Pre{type(self).__name__}.angle_k1"]
        k2 = ff_params[f"Pre{type(self).__name__}.angle_k2"]
        b1, b2 = PreMMBondedConj.angle_b1, PreMMBondedConj.angle_b2
        coords = graph.coords
        node_idx = graph.inc_node_angle.long()
        counts = graph.get_count("angle", idx=None, cluster=cluster)
        return CFF.calc_conj_angle(coords, k1, k2, b1, b2, node_idx, counts, calc_partial_hessian)
