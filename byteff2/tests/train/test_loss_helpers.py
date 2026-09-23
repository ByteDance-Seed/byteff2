# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from byteff2.train import loss as loss_mod
from byteff2.train.loss import calc_conf_mean, loss_func, LossType, soft_mse
from byteff2.utils.definitions import MMParam, MMTerm, MMTERM_WIDTH


class DummyData(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        self._counts = kwargs.get("_counts", {})

    def get_count(self, key, *_args, **_kwargs):
        return self._counts.get(key, torch.tensor([0], dtype=torch.long))


def test_soft_mse_keep_dim_matches_mean_and_handles_zero():
    diff = torch.tensor([0.0, 1.0, -4.0], dtype=torch.float64)
    keep_dim = soft_mse(diff, max_val=2.0, keep_dim=True)
    reduced = soft_mse(diff, max_val=2.0, keep_dim=False)

    assert keep_dim.shape == diff.shape
    assert torch.isfinite(keep_dim).all()
    assert keep_dim[0].item() == pytest.approx(0.0)
    assert reduced.item() == pytest.approx(keep_dim.mean().item())


def test_calc_conf_mean_applies_masked_average():
    src = torch.tensor([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]], dtype=torch.float64)
    confmask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float64)

    result = calc_conf_mean(src, confmask)

    expected = torch.tensor([(2.0 + 4.0) / 2.0, (7.0 + 9.0) / 2.0], dtype=torch.float64)
    assert torch.allclose(result, expected)


def test_loss_func_param_mse_uses_named_prediction_and_label():
    preds = {"ff_parameters": {"demo_param": torch.tensor([[1.0], [3.0]], dtype=torch.float64)}}
    data = {"demo_label": torch.tensor([[1.0], [5.0]], dtype=torch.float64)}

    loss = loss_func(preds, data, LossType.ParamMSE, label="demo_label", param="demo_param")

    assert loss.item() == pytest.approx(2.0)


def test_loss_func_bonded_energy_returns_zero_tensor_when_term_is_empty(monkeypatch):
    monkeypatch.setattr(loss_mod, "CFF", SimpleNamespace(calc_bond=lambda *_args, **_kwargs: (0.0, None, None)))

    preds = {
        "ff_parameters": {
            "PreMMBonded.bond_k": torch.tensor([[1.0]], dtype=torch.float64),
            "PreMMBonded.bond_r0": torch.tensor([[1.0]], dtype=torch.float64),
        }
    }
    data = DummyData(
        coords=torch.zeros((1, 1, 3), dtype=torch.float64),
        inc_node_bond=torch.zeros((0, 2), dtype=torch.long),
    )

    loss = loss_func(preds, data, LossType.BondedEnergy, param="bond")

    assert torch.is_tensor(loss)
    assert loss.dtype == data.coords.dtype
    assert loss.device == data.coords.device
    assert loss.item() == pytest.approx(0.0)


def test_loss_func_inter_energy_mse_uses_plain_masked_mean():
    preds = {
        "energy": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        "energy_cluster": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    }
    data = DummyData(
        name=["demo_LI_cluster"],
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        confmask_cluster=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        total_int_energy=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        coords=torch.zeros((1, 2, 3), dtype=torch.float64),
        _counts={
            "node": torch.tensor([1], dtype=torch.long),
            "mol": torch.tensor([1], dtype=torch.long),
        },
    )

    loss = loss_func(preds, data, LossType.InterEnergyMSE)

    assert loss.item() == pytest.approx(2.5)


def test_loss_func_inter_energy_disp_mse_supports_distance_scaling():
    preds = {"ff_parameters": {"DISP": torch.tensor([[2.0, 3.0]], dtype=torch.float64)}}
    data = DummyData(
        name=["demo"],
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        confmask_cluster=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        disp_int_energy=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        min_dists=torch.tensor([[2.0, 3.0]], dtype=torch.float64),
        coords=torch.zeros((1, 2, 3), dtype=torch.float64),
        _counts={"node": torch.tensor([1], dtype=torch.long)},
    )

    loss = loss_func(
        preds,
        data,
        LossType.InterEnergyDispMSE,
        scale_by_min_dist=True,
    )

    assert loss.item() == pytest.approx(20.0)


def test_loss_func_inter_energy_ct_and_elec_pauli_mse_paths():
    data = DummyData(
        name=["demo"],
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        confmask_cluster=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        charge_transfer_int_energy=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        elec_pauli_int_energy=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        coords=torch.zeros((1, 2, 3), dtype=torch.float64),
        _counts={"node": torch.tensor([1], dtype=torch.long)},
    )
    preds = {
        "ff_parameters": {
            "CHARGE_TRANSFER": torch.tensor([[2.0, 4.0]], dtype=torch.float64),
            "ELEC": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            "PAULI": torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        }
    }

    ct_loss = loss_func(preds, data, LossType.InterEnergyCTMSE)
    elec_pauli_loss = loss_func(preds, data, LossType.InterEnergyElecPauliMSE)

    assert ct_loss.item() == pytest.approx(5.0)
    assert elec_pauli_loss.item() == pytest.approx(2.5)


def test_loss_func_partial_hessian_mape_masks_large_entries(monkeypatch):
    def fake_energy_force(*_args, **_kwargs):
        results = {}
        for term in MMTerm:
            if term is MMTerm.bond:
                width = MMTERM_WIDTH[term]
                hessian = torch.zeros((1, 1, width * width, 9), dtype=torch.float64)
                hessian[:, :, 1, :] = 2.0e5
                hessian[:, :, 2, :] = 2.0e5
                results[term.name] = (None, None, hessian)
            else:
                results[term.name] = (None, None, None)
        return results

    monkeypatch.setattr(loss_mod, "CFF", SimpleNamespace(energy_force=fake_energy_force))

    payload = {
        "partial_hessian": torch.zeros((1, 1, 9), dtype=torch.float64),
        "coords": torch.zeros((4, 1, 3), dtype=torch.float64),
    }
    counts = {
        "partial_hessian": torch.tensor([1], dtype=torch.long),
        "bond": torch.tensor([1], dtype=torch.long),
        "angle": torch.tensor([0], dtype=torch.long),
        "proper": torch.tensor([0], dtype=torch.long),
        "improper": torch.tensor([0], dtype=torch.long),
    }
    for term in MMTerm:
        width = MMTERM_WIDTH[term]
        for i in range(width):
            for j in range(width):
                payload[f"{term.name}_rec_{i}_{j}"] = torch.zeros(
                    (max(counts[term.name].sum().item(), 1),), dtype=torch.long
                )
    data = DummyData(**payload, _counts=counts)
    preds = {
        "ff_parameters": {
            "PreMMBonded.bond_k": torch.ones((1, 1), dtype=torch.float64),
            "PreMMBonded.bond_r0": torch.ones((1, 1), dtype=torch.float64),
            "PreMMBonded.angle_k": torch.ones((1, 1), dtype=torch.float64),
            "PreMMBonded.angle_d0": torch.ones((1, 1), dtype=torch.float64),
            "PreMMBonded.proper_k": torch.ones((1, 4), dtype=torch.float64),
            "PreMMBonded.improper_k": torch.ones((1, 1), dtype=torch.float64),
        }
    }

    loss = loss_func(preds, data, LossType.Partial_Hessian_MAPE, mask_threshold=1.0e4)

    assert loss.item() == pytest.approx(0.0)


def test_loss_func_boltzmann_soft_mse_and_l1_norm_support_circstd_mask(monkeypatch):
    monkeypatch.setattr(
        loss_mod,
        "get_dihedral_angle_vec",
        lambda *_args, **_kwargs: (torch.zeros((1, 2), dtype=torch.float64), None, None, None),
    )

    def fake_energy_force(*_args, **_kwargs):
        return {"proper": (torch.tensor([[0.5, 0.5]], dtype=torch.float64), None, None)}

    monkeypatch.setattr(loss_mod, "CFF", SimpleNamespace(energy_force=fake_energy_force))

    data = DummyData(
        coords=torch.zeros((4, 2, 3), dtype=torch.float64),
        inc_node_proper=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        energy=torch.tensor([[1.5, 1.0]], dtype=torch.float64),
        _counts={
            "node": torch.tensor([4], dtype=torch.long),
            "proper": torch.tensor([1], dtype=torch.long),
        },
    )
    preds = {
        "energy": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        "ff_parameters": {
            "PreMMBonded.proper_k": torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float64),
            "demo_param": torch.tensor([[1.0], [-3.0]], dtype=torch.float64),
        },
    }

    boltzmann_loss = loss_func(
        preds,
        data,
        LossType.Boltzman_Soft_MSE,
        clamp=2.0,
        decay=2.0,
        max=100.0,
        mask_by_circstd_threshold=10.0,
    )
    l1_loss = loss_func(
        preds,
        data,
        LossType.L1_Norm,
        param="demo_param",
        mask_by_circstd_threshold=10.0,
    )

    assert torch.isfinite(boltzmann_loss)
    assert boltzmann_loss.item() > 0.0
    assert l1_loss.item() == pytest.approx(2.0)


def test_loss_func_boltzmann_soft_mse_reuses_dist_scale_and_force_cutoff(monkeypatch):
    monkeypatch.setattr(
        loss_mod,
        "get_dihedral_angle_vec",
        lambda *_args, **_kwargs: (torch.zeros((1, 2), dtype=torch.float64), None, None, None),
    )
    monkeypatch.setattr(loss_mod, "reduce_counts", lambda x, _counts, reduce="max": x)

    cutoff_calls = []

    def fake_cosine_cutoff(x, left, right):
        cutoff_calls.append((x.clone(), left, right))
        if len(cutoff_calls) == 1:
            return torch.full_like(x, 0.25)
        return torch.full_like(x, 0.5)

    def fake_energy_force(*_args, **_kwargs):
        return {"proper": (torch.tensor([[0.5, 0.5]], dtype=torch.float64), None, None)}

    monkeypatch.setattr(loss_mod, "cosine_cutoff", fake_cosine_cutoff)
    monkeypatch.setattr(loss_mod, "CFF", SimpleNamespace(energy_force=fake_energy_force))

    data = DummyData(
        coords=torch.zeros((4, 2, 3), dtype=torch.float64),
        inc_node_proper=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        min_dists=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        energy=torch.tensor([[1.5, 1.0]], dtype=torch.float64),
        _counts={
            "node": torch.tensor([1], dtype=torch.long),
            "proper": torch.tensor([1], dtype=torch.long),
        },
    )
    preds = {
        "energy": torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        "forces": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        "forces_cluster": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        "ff_parameters": {
            "PreMMBonded.proper_k": torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float64),
        },
    }

    loss = loss_func(
        preds,
        data,
        LossType.Boltzman_Soft_MSE,
        clamp=2.0,
        decay=2.0,
        max=100.0,
        mask_by_circstd_threshold=10.0,
        dist_scale=(0.0, 1.0),
        force_cutoff=(0.0, 1.0),
    )

    assert torch.isfinite(loss)
    assert len(cutoff_calls) == 4
    assert torch.allclose(cutoff_calls[0][0], data.min_dists)
    assert torch.allclose(cutoff_calls[2][0], data.min_dists)
    assert torch.allclose(cutoff_calls[1][0], cutoff_calls[3][0])
    assert torch.allclose(cutoff_calls[1][0], torch.tensor([1.0], dtype=torch.float64))


def test_loss_func_mm_bonded_conj_mse_matches_constructed_targets(monkeypatch):
    monkeypatch.setattr(
        loss_mod,
        "PreMMBondedConj",
        SimpleNamespace(
            param_std_mean_range={
                "bond_k1": (2.0,),
                "bond_k2": (2.0,),
                "angle_k1": (3.0,),
                "angle_k2": (3.0,),
                "torsion": (1.0,),
            },
            bond_b1=1.0,
            bond_b2=3.0,
            angle_b1=0.0,
            angle_b2=torch.pi,
        ),
    )
    data = DummyData(
        bond_k=torch.tensor([[4.0]], dtype=torch.float64),
        bond_r0=torch.tensor([[2.0]], dtype=torch.float64),
        angle_k=torch.tensor([[6.0]], dtype=torch.float64),
        angle_d0=torch.tensor([[90.0]], dtype=torch.float64),
        torsion=torch.tensor([[1.5]], dtype=torch.float64),
    )
    preds = {
        "ff_parameters": {
            "PreMMBondedConj.bond_k1": torch.tensor([[2.0]], dtype=torch.float64),
            "PreMMBondedConj.bond_k2": torch.tensor([[2.0]], dtype=torch.float64),
            "PreMMBondedConj.angle_k1": torch.tensor([[3.0]], dtype=torch.float64),
            "PreMMBondedConj.angle_k2": torch.tensor([[3.0]], dtype=torch.float64),
            "PreMMBondedConj.torsion": torch.tensor([[1.5]], dtype=torch.float64),
        }
    }

    loss = loss_func(preds, data, LossType.MMBondedConjMSE)

    assert loss.item() == pytest.approx(0.0)


def test_loss_func_bonded_energy_angle_improper_and_invalid_param(monkeypatch):
    monkeypatch.setattr(
        loss_mod,
        "CFF",
        SimpleNamespace(
            calc_angle=lambda *_args, **_kwargs: (torch.tensor([6.0], dtype=torch.float64), None, None),
            calc_improper=lambda *_args, **_kwargs: (torch.tensor([8.0], dtype=torch.float64), None, None),
        ),
    )
    preds = {
        "ff_parameters": {
            "PreMMBonded.angle_k": torch.tensor([[1.0]], dtype=torch.float64),
            "PreMMBonded.angle_d0": torch.tensor([[2.0]], dtype=torch.float64),
            "PreMMBonded.improper_k": torch.tensor([[3.0]], dtype=torch.float64),
        }
    }
    data = DummyData(
        coords=torch.zeros((1, 1, 3), dtype=torch.float64),
        inc_node_angle=torch.zeros((1, 3), dtype=torch.long),
        inc_node_improper=torch.zeros((1, 4), dtype=torch.long),
        _counts={"angle": torch.tensor([2], dtype=torch.long), "improper": torch.tensor([4], dtype=torch.long)},
    )

    angle_loss = loss_func(preds, data, LossType.BondedEnergy, param="angle")
    improper_loss = loss_func(preds, data, LossType.BondedEnergy, param="improper")

    assert angle_loss.item() == pytest.approx(3.0)
    assert improper_loss.item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Unsupported trained_param_name"):
        loss_func(preds, data, LossType.BondedEnergy, param="torsion")


def test_loss_func_inter_energy_mse_supports_force_cutoff_dist_scale_and_boltzmann(monkeypatch):
    monkeypatch.setattr(loss_mod, "cosine_cutoff", lambda x, *_args: torch.full_like(x, 0.5))
    monkeypatch.setattr(loss_mod, "reduce_counts", lambda x, _counts, reduce="max": x)

    preds = {
        "energy": torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        "energy_cluster": torch.tensor([[3.0, 4.0]], dtype=torch.float64),
        "forces": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        "forces_cluster": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    }
    data = DummyData(
        name=["demo"],
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        confmask_cluster=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        total_int_energy=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        min_dists=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        coords=torch.zeros((1, 2, 3), dtype=torch.float64),
        _counts={"node": torch.tensor([1], dtype=torch.long), "mol": torch.tensor([1], dtype=torch.long)},
    )

    loss = loss_func(
        preds,
        data,
        LossType.InterEnergyMSE,
        force_cutoff=(0.0, 1.0),
        dist_scale=(0.0, 1.0),
        clamp=2.0,
        decay=2.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_loss_func_inter_energy_pol_mse_supports_boltzmann_weighting():
    preds = {
        "energy": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        "energy_cluster": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        "ff_parameters": {"POLARIZATION": torch.tensor([[2.0, 4.0]], dtype=torch.float64)},
    }
    data = DummyData(
        name=["PF6_LI_demo"],
        confmask=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        confmask_cluster=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        polarization_int_energy=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        total_int_energy=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        coords=torch.zeros((1, 2, 3), dtype=torch.float64),
        _counts={"node": torch.tensor([1], dtype=torch.long), "mol": torch.tensor([1], dtype=torch.long)},
    )

    loss = loss_func(
        preds,
        data,
        LossType.InterEnergyPolMSE,
        clamp=2.0,
        decay=2.0,
    )

    assert loss.item() == pytest.approx(5.0)


def test_loss_func_raises_for_unused_kwargs():
    preds = {"ff_parameters": {"demo_param": torch.tensor([[1.0], [3.0]], dtype=torch.float64)}}
    data = {"demo_label": torch.tensor([[1.0], [5.0]], dtype=torch.float64)}

    with pytest.raises(ValueError, match="Unused kwargs for ParamMSE: unexpected"):
        loss_func(
            preds,
            data,
            LossType.ParamMSE,
            label="demo_label",
            param="demo_param",
            unexpected=True,
        )


def test_loss_func_raises_for_unknown_loss_type():
    preds = {"ff_parameters": {f"PreMMBonded.{MMParam.bond_k.name}": torch.tensor([[1.0]])}}
    data = DummyData(coords=torch.zeros((1, 1, 3)))

    with pytest.raises(NotImplementedError):
        loss_func(preds, data, object())
