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
from torch.utils.data import DataLoader

from byteff2.data import collate_data, MonoData
from byteff2.model.ff_layers import MMBonded, PreMMBonded
from byteff2.tests.data.test_dataset import create_hessian_dataset, create_mono_dataset
from byteff2.train.loss import loss_func, LossType


def test_hessian_loss(tmp_path):

    dim = 64
    premm_layer = PreMMBonded(dim, dim)
    _ = MMBonded(dim, dim)

    dataset = create_hessian_dataset()
    dataloader = DataLoader(dataset, batch_size=40, shuffle=False, collate_fn=collate_data)
    data: MonoData = next(iter(dataloader))

    # partial_hessian label shape: [nPartialHessian, nconfs, 9]
    partial_hessian = data["partial_hessian"]
    assert partial_hessian.dim() == 3 and partial_hessian.shape[-1] == 9

    x_h, e_h = torch.rand((data.node_features.shape[0], dim)), torch.rand((data.edge_features.shape[0], dim))
    ff_params = premm_layer(data, x_h, e_h)
    loss = loss_func(
        {"ff_parameters": ff_params},
        data,
        LossType.Partial_Hessian_MAPE,
    )

    # loss should be a scalar finite tensor, and strictly positive for a random init
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    # ensure the gradient chain to trainable params (bond_k / angle_k / improper_k)
    # is not accidentally detached
    assert loss.requires_grad
    assert loss.detach() > 0


def test_energy_force_loss(tmp_path):

    _, dataset = create_mono_dataset(tmp_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_data)
    data: MonoData = next(iter(dataloader))

    n_node, n_conf = data.coords.shape[0], data.coords.shape[1]
    n_mol = data.get_count("node", idx=None).shape[0]

    # build a differentiable prediction whose gradient chain reaches a trainable tensor
    theta = torch.zeros((), requires_grad=True)
    pred_energy = data["energy"] + theta
    pred_forces = data["forces"] + theta
    preds = {"energy": pred_energy, "forces": pred_forces, "ff_parameters": {}}

    assert pred_energy.shape == (n_mol, n_conf)
    assert pred_forces.shape == (n_node, n_conf, 3)

    for lt in (LossType.Energy_MSE, LossType.Energy_Soft_MSE, LossType.Force_MSE, LossType.Force_Soft_MSE):
        loss = loss_func(preds, data, lt)
        assert loss.shape == torch.Size([]), lt
        assert torch.isfinite(loss), lt
        assert loss.requires_grad, lt
        # perfect prediction (shift removes constant offset) -> zero loss, but grad chain intact
        assert loss.detach() >= 0, lt

    # a nonzero prediction error must give strictly positive loss
    preds_bad = {
        "energy": data["energy"] + torch.arange(n_conf, dtype=data["energy"].dtype),
        "forces": data["forces"] + 1.0,
        "ff_parameters": {},
    }
    assert loss_func(preds_bad, data, LossType.Energy_MSE).detach() > 0
    assert loss_func(preds_bad, data, LossType.Force_MSE).detach() > 0
