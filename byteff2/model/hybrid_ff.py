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

from collections import defaultdict
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from byteff2.data import Data
from byteff2.model import ff_layers

from .graph_block import Graph2DBlock


class PreForceField(Module):
    def __init__(self, node_dim: int, edge_dim: int, configs: list[dict]) -> None:
        super().__init__()

        self.pre_layers: dict[str, ff_layers.PreFFLayer] = ModuleDict()
        for conf in configs:
            layer_type = "Pre" + conf["type"]
            assert layer_type not in self.pre_layers
            layer_cls = getattr(ff_layers, layer_type)
            layer = layer_cls(node_dim, edge_dim, **conf)
            self.pre_layers[layer_type] = layer

    def reset_parameters(self):
        for layer in self.pre_layers.values():
            layer.reset_parameters()

    def get_parameters(self, layer_type: str):
        return self.pre_layers["Pre" + layer_type].parameters()

    def forward(
        self, data: Data, x_h: Tensor, e_h: Tensor, ff_parameters: dict, do_patch: bool = False
    ) -> dict[str, Tensor]:

        for layer in self.pre_layers.values():
            if isinstance(layer, ff_layers.PreMMBonded):
                ff_parameters.update(layer(data, x_h, e_h, ff_parameters, do_patch=do_patch))
            else:
                ff_parameters.update(layer(data, x_h, e_h, ff_parameters))

        return ff_parameters


class ForceField(Module):
    def __init__(self, node_dim: int, edge_dim: int, configs: list[dict]) -> None:
        super().__init__()
        self.ff_require_grad = {}
        self.ff_layers: dict[str, ff_layers.FFLayer] = ModuleDict()
        for conf in configs:
            layer_type = conf["type"]
            assert layer_type not in self.ff_layers
            layer_cls = getattr(ff_layers, layer_type)
            layer = layer_cls(node_dim, edge_dim, **conf)
            self.ff_layers[layer_type] = layer
            self.ff_require_grad[layer_type] = conf.get("ff_require_grad", True)

    def get_parameters(self, layer_type: str):
        return self.ff_layers[layer_type].parameters()

    def reset_parameters(self):
        for layer in self.ff_layers.values():
            layer.reset_parameters()

    def forward(
        self,
        data: Data,
        x_h: Tensor,
        e_h: Tensor,
        ff_parameters: dict[str, Tensor] = None,
        cluster: bool = False,
        skip_ff: Union[bool, list[str]] = False,
    ) -> tuple[Tensor, Tensor]:

        tot_energy, tot_forces = (
            0.0,
            0,
        )
        if skip_ff is True:
            return tot_energy, tot_forces

        for layer_type, layer in self.ff_layers.items():
            if skip_ff and layer_type in skip_ff:
                continue
            energy, forces = layer(data, x_h, e_h, ff_parameters, cluster)
            if not self.ff_require_grad[layer_type]:
                energy = energy.detach().clone()
                forces = forces.detach().clone()
            tot_energy += energy
            tot_forces += forces

            suffix = "_cluster" if cluster else ""
            ff_parameters[f"{layer_type}.energy{suffix}"] = energy
            ff_parameters[f"{layer_type}.forces{suffix}"] = forces

        return tot_energy, tot_forces


class HybridFF(Module):
    def __init__(
        self,
        graph_block: dict,
        ff_block: list[dict],
        supported_elements: list[int],
    ):
        super().__init__()

        self.supported_elements = tuple(supported_elements)
        self.graph_block = Graph2DBlock(**graph_block)
        self.preff_block = PreForceField(self.graph_block.node_out_dim, self.graph_block.edge_out_dim, ff_block)
        self.ff_block = ForceField(self.graph_block.node_out_dim, self.graph_block.edge_out_dim, ff_block)
        self.reset_parameters()

    def reset_parameters(self):
        self.graph_block.reset_parameters()
        self.preff_block.reset_parameters()
        self.ff_block.reset_parameters()

    def get_parameters(self, name=None):
        if name is None:
            return self.parameters()
        elif name == "Graph":
            return self.graph_block.parameters()
        else:
            return list(self.preff_block.get_parameters(name)) + list(self.ff_block.get_parameters(name))

    def _validate_supported_elements(self, data: Data) -> None:
        if self.training:
            return

        node_features = data.node_features
        atomic_numbers = node_features[:, 0].long() + 1
        supported = set(self.supported_elements)
        unsupported = sorted({int(z.item()) for z in atomic_numbers if int(z.item()) not in supported})
        if unsupported:
            supported_str = ", ".join(str(z) for z in self.supported_elements)
            unsupported_str = ", ".join(str(z) for z in unsupported)
            raise ValueError(
                "Input contains unsupported atomic numbers in eval mode: "
                f"{unsupported_str}. Supported atomic numbers: [{supported_str}]"
            )

    def forward(
        self,
        data: Data,
        cluster=False,
        skip_ff=False,
        do_patch: bool = False,
        validate_elements: bool = True,
    ):
        if validate_elements:
            self._validate_supported_elements(data)
        node_h, edge_h, xs = self.graph_block(data)
        ff_parameters = {"Graph2D.xs": xs}
        ff_parameters = self.preff_block(data, node_h, edge_h, ff_parameters, do_patch=do_patch)
        energy, forces = self.ff_block(data, node_h, edge_h, ff_parameters, cluster=False, skip_ff=skip_ff)
        preds = {"ff_parameters": ff_parameters, "energy": energy, "forces": forces}
        if cluster:
            energy_cluster, forces_cluster = self.ff_block(
                data, node_h, edge_h, ff_parameters, cluster=True, skip_ff=skip_ff
            )
            preds["energy_cluster"] = energy_cluster
            preds["forces_cluster"] = forces_cluster
        return preds


class EnsembleModel:
    def __init__(self, models: list[HybridFF]):
        self.models = models

        self.std = None
        self.rstd = None

    @torch.no_grad()
    def __call__(self, *args, validate_elements: bool = True, **kwds):
        params = defaultdict(list)
        for model in self.models:
            preds = model(*args, validate_elements=validate_elements, **kwds)
            for k, v in preds["ff_parameters"].items():
                if k == "Graph2D.xs":
                    continue
                params[k].append(v)

        for k, v in params.items():
            ps = torch.stack(v, dim=0)
            mean_ps = ps.mean(dim=0)
            std_ps = ps.std(dim=0)
            rstd_ps = std_ps / torch.where(mean_ps > 1.0, mean_ps, 1.0)
            preds["ff_parameters"][k] = mean_ps
            preds["ff_parameters"].setdefault("uncertainty", {})[k] = rstd_ps
        return preds

    def eval(self):
        for model in self.models:
            model.eval()
