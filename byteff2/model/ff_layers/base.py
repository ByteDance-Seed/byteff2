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

from torch import Tensor
from torch.nn import Module

from byteff2.data import Data


class PreFFLayer(Module):
    """Base class for PreFFLayer, 
        infering coordinate-invariant parameters used in FFLayer."""

    def __init__(self, node_dim: int, edge_dim: int, **configs):
        super().__init__()

    def reset_parameters(self):
        raise NotImplementedError()

    def forward(self, data: Data, x_h: Tensor, e_h: Tensor, ff_parameters: dict[str, Tensor] = None):
        raise NotImplementedError()


class FFLayer(Module):
    """Base class for FFLayer, 
        infering energy given coordinates."""

    def __init__(self, node_dim: int, edge_dim: int, **configs):
        super().__init__()

    def reset_parameters(self):
        raise NotImplementedError()

    def forward(self, data: Data, x_h: Tensor, e_h: Tensor, ff_parameters: dict[str, Tensor], cluster: bool = False):
        raise NotImplementedError()
