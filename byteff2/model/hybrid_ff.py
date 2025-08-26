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

from typing import Union

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
            layer_type = 'Pre' + conf['type']
            assert layer_type not in self.pre_layers
            layer_cls = getattr(ff_layers, layer_type)
            layer = layer_cls(node_dim, edge_dim, **conf)
            self.pre_layers[layer_type] = layer

    def reset_parameters(self):
        for layer in self.pre_layers.values():
            layer.reset_parameters()

    def get_parameters(self, layer_type: str):
        return self.pre_layers['Pre' + layer_type].parameters()

    def forward(self, data: Data, x_h: Tensor, e_h: Tensor, ff_parameters: dict) -> dict[str, Tensor]:

        for layer in self.pre_layers.values():
            ff_parameters.update(layer(data, x_h, e_h, ff_parameters))

        return ff_parameters


class ForceField(Module):

    def __init__(self, node_dim: int, edge_dim: int, configs: list[dict]) -> None:
        super().__init__()
        self.ff_require_grad = {}
        self.ff_layers: dict[str, ff_layers.FFLayer] = ModuleDict()
        for conf in configs:
            layer_type = conf['type']
            assert layer_type not in self.ff_layers
            layer_cls = getattr(ff_layers, layer_type)
            layer = layer_cls(node_dim, edge_dim, **conf)
            self.ff_layers[layer_type] = layer
            self.ff_require_grad[layer_type] = conf.get('ff_require_grad', True)

    def get_parameters(self, layer_type: str):
        return self.ff_layers[layer_type].parameters()

    def reset_parameters(self):
        for layer in self.ff_layers.values():
            layer.reset_parameters()

    def forward(self,
                data: Data,
                x_h: Tensor,
                e_h: Tensor,
                ff_parameters: dict[str, Tensor] = None,
                cluster: bool = False,
                skip_ff: Union[bool, list[str]] = False) -> tuple[Tensor, Tensor]:

        tot_energy, tot_forces = 0., 0,
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

            suffix = '_cluster' if cluster else ''
            ff_parameters[f'{layer_type}.energy{suffix}'] = energy
            ff_parameters[f'{layer_type}.forces{suffix}'] = forces

        return tot_energy, tot_forces


class HybridFF(Module):

    def __init__(self, graph_block: dict, ff_block: list[dict]):
        super().__init__()

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
        elif name == 'Graph':
            return self.graph_block.parameters()
        else:
            return list(self.preff_block.get_parameters(name)) + list(self.ff_block.get_parameters(name))

    def forward(self, data: Data, cluster=False, skip_ff=False):
        node_h, edge_h, xs = self.graph_block(data)
        ff_parameters = {'Graph2D.xs': xs}
        ff_parameters = self.preff_block(data, node_h, edge_h, ff_parameters)
        energy, forces = self.ff_block(data, node_h, edge_h, ff_parameters, cluster=False, skip_ff=skip_ff)
        preds = {'ff_parameters': ff_parameters, 'energy': energy, 'forces': forces}
        if cluster:
            energy_cluster, forces_cluster = self.ff_block(data,
                                                           node_h,
                                                           edge_h,
                                                           ff_parameters,
                                                           cluster=True,
                                                           skip_ff=skip_ff)
            preds['energy_cluster'] = energy_cluster
            preds['forces_cluster'] = forces_cluster
        return preds
