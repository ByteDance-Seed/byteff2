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

"""Tests for byteff2.train.trainer._valid_needs_grad.

Regression coverage for the validation autograd-context decision used in
`FFTrainer.valid_epoch`. Specifically:
  - Vanilla datasets with ordinary losses should run under
    `torch.inference_mode()`.
  - Datasets configured with autograd-based losses must keep autograd enabled.
  - The inter-energy losses are explicitly treated as autograd-required in the
    current implementation and should stay on that whitelist.
"""

import pytest

from byteff2.train.trainer import _AUTOGRAD_REQUIRED_LOSSES, _valid_needs_grad


def _ds(loss=None, aux_loss=None, **extra):
    """Build a minimal dataset config dict for the helper under test."""
    cfg = {"loss": loss or [{"loss_type": "ParamMSE", "weight": 1.0}]}
    if aux_loss is not None:
        cfg["aux_loss"] = aux_loss
    cfg.update(extra)
    return cfg


# --- non-grad paths ---------------------------------------------------------


def test_vanilla_dataset_does_not_need_grad():
    """Plain non-polarizable dataset with non-autograd losses -> can use
    torch.inference_mode() during validation."""
    assert _valid_needs_grad(_ds()) is False


def test_multiple_non_autograd_losses_does_not_need_grad():
    cfg = _ds(
        loss=[
            {"loss_type": "ParamMSE", "weight": 1.0},
            {"loss_type": "BondedEnergy", "weight": 1.0},
        ]
    )
    assert _valid_needs_grad(cfg) is False


def test_aux_loss_only_non_autograd_does_not_need_grad():
    cfg = _ds(aux_loss=[{"loss_type": "L1_Norm"}])
    assert _valid_needs_grad(cfg) is False


def test_explicit_false_flags_does_not_need_grad():
    cfg = _ds(cluster=False, record_D_ind=False)
    assert _valid_needs_grad(cfg) is False


# --- autograd-required loss types ------------------------------------------


def test_partial_hessian_mape_in_main_loss_needs_grad():
    cfg = _ds(loss=[{"loss_type": "Partial_Hessian_MAPE", "weight": 1.0}])
    assert _valid_needs_grad(cfg) is True


def test_partial_hessian_mape_in_aux_loss_needs_grad():
    """Autograd-based loss can also appear as an aux_loss."""
    cfg = _ds(aux_loss=[{"loss_type": "Partial_Hessian_MAPE"}])
    assert _valid_needs_grad(cfg) is True


def test_autograd_required_losses_set_includes_partial_hessian():
    """Whitelist must always contain Partial_Hessian_MAPE; otherwise the
    autograd-based hessian validation path silently breaks."""
    assert "Partial_Hessian_MAPE" in _AUTOGRAD_REQUIRED_LOSSES


# --- autograd-required inter-energy losses ---------------------------------


@pytest.mark.parametrize(
    "loss_name",
    [
        "InterEnergyMSE",
        "InterEnergyPolMSE",
        "InterEnergyDispMSE",
        "InterEnergyCTMSE",
        "InterEnergyElecPauliMSE",
    ],
)
def test_inter_energy_losses_need_grad(loss_name):
    cfg = _ds(loss=[{"loss_type": loss_name, "weight": 1.0}])
    assert _valid_needs_grad(cfg) is True


def test_cluster_flag_alone_does_not_force_grad():
    cfg = _ds(loss=[{"loss_type": "ParamMSE", "weight": 1.0}], cluster=True)
    assert _valid_needs_grad(cfg) is False


def test_record_d_ind_flag_alone_does_not_force_grad():
    cfg = _ds(record_D_ind=True)
    assert _valid_needs_grad(cfg) is False


# --- robustness -------------------------------------------------------------


def test_missing_loss_key_does_not_crash():
    """`get('loss', []) or []` should also tolerate an absent or None value."""
    assert _valid_needs_grad({}) is False
    assert _valid_needs_grad({"loss": None}) is False
    assert _valid_needs_grad({"loss": None, "aux_loss": None}) is False


def test_empty_loss_list_does_not_need_grad():
    assert _valid_needs_grad({"loss": []}) is False
