# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

"""Numerical regression tests for the released ByteFF-26 weights."""

import pytest
import torch
import yaml

from byteff2.data import GraphData
from byteff2.train import load_pretrained_model
from byteff2.utils import get_asset_path


_SMILES = {
    "methanol": "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
    "ethanol": "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
}
# Captured on CPU/float32 before stripping training metadata from assets
# github-public@4a9278c1. Members and weights are unchanged by that cleanup.
# Bond lengths/sigma: Angstrom; charges: e; epsilon/proper coefficients: kcal/mol.
_EXPECTED = {
    "member_01": {
        "methanol": {
            "PreMMBonded.bond_r0": [1.427439, 1.097327, 1.097327, 1.097327, 0.970934],
            "PreLJEs.charges": [0.106954, -0.610044, 0.035377, 0.035377, 0.035377, 0.396961],
            "PreLJEs.sigma": [3.401453, 3.242332, 2.421583, 2.421583, 2.421585, 0.543232],
            "PreLJEs.epsilon": [0.108805, 0.093386, 0.020479, 0.020479, 0.020479, 0.005400],
            "proper_first": [2.138886, -0.727873, 0.129636, 0.085198],
        },
        "ethanol": {
            "PreMMBonded.bond_r0": [1.538630, 1.097402, 1.097402, 1.097402, 1.439168, 1.099172, 1.099172, 0.971154],
            "PreLJEs.charges": [
                -0.127867,
                0.130219,
                -0.601422,
                0.043047,
                0.043047,
                0.043047,
                0.036772,
                0.036772,
                0.396385,
            ],
            "PreLJEs.sigma": [3.398520, 3.396366, 3.241568, 2.599907, 2.599907, 2.599907, 2.421400, 2.421400, 0.538729],
            "PreLJEs.epsilon": [
                0.107801,
                0.109558,
                0.092447,
                0.020729,
                0.020729,
                0.020729,
                0.020779,
                0.020779,
                0.005190,
            ],
            "proper_first": [2.754592, -0.614106, 0.224113, 0.114914],
        },
    },
    "ensemble": {
        "methanol": {
            "PreMMBonded.bond_r0": [1.430676, 1.097721, 1.097721, 1.097721, 0.970551],
            "PreLJEs.charges": [0.101584, -0.607608, 0.034342, 0.034342, 0.034342, 0.403000],
            "PreLJEs.sigma": [3.398123, 3.244343, 2.423664, 2.423663, 2.423664, 0.541269],
            "PreLJEs.epsilon": [0.108265, 0.093721, 0.020768, 0.020768, 0.020768, 0.004907],
            "proper_first": [2.066701, -0.699462, 0.176702, 0.090211],
        },
        "ethanol": {
            "PreMMBonded.bond_r0": [1.537883, 1.097263, 1.097263, 1.097263, 1.439709, 1.098940, 1.098940, 0.970673],
            "PreLJEs.charges": [
                -0.121776,
                0.123877,
                -0.600854,
                0.042134,
                0.042134,
                0.042134,
                0.037254,
                0.037254,
                0.397841,
            ],
            "PreLJEs.sigma": [3.397170, 3.398046, 3.241250, 2.600077, 2.600077, 2.600077, 2.422850, 2.422850, 0.538088],
            "PreLJEs.epsilon": [
                0.107718,
                0.108578,
                0.093660,
                0.020818,
                0.020818,
                0.020818,
                0.020788,
                0.020788,
                0.004823,
            ],
            "proper_first": [2.859027, -0.618626, 0.214438, 0.136199],
        },
    },
}
# Allow ordinary floating-point variation without accepting wrong weights/units.
_ATOL = {
    "PreMMBonded.bond_r0": 0.002,
    "PreLJEs.charges": 0.0005,
    "PreLJEs.sigma": 0.005,
    "PreLJEs.epsilon": 0.0005,
    "proper_first": 0.005,
}


@pytest.fixture(scope="module")
def models():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield {
            "member_01": load_pretrained_model("ByteFF-26", member=1, device="cpu"),
            "ensemble": load_pretrained_model("ByteFF-26", device="cpu"),
        }
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("model_name", ["member_01", "ensemble"])
@pytest.mark.parametrize("molecule_name", ["methanol", "ethanol"])
def test_pretrained_byteff26_parameters(models, model_name, molecule_name):
    model = models[model_name]
    if model_name == "ensemble":
        assert len(model.models) == 5
    data = GraphData(molecule_name, _SMILES[molecule_name])
    with torch.no_grad():
        parameters = model(data, skip_ff=True, do_patch=False)["ff_parameters"]
    for key, values in _EXPECTED[model_name][molecule_name].items():
        actual = parameters["PreMMBonded.proper_k"][0] if key == "proper_first" else parameters[key].flatten()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, actual.new_tensor(values), rtol=0.002, atol=_ATOL[key])
    assert abs(parameters["PreLJEs.charges"].sum().item()) < 1e-4


def test_pretrained_byteff26_member_energy_and_forces(models):
    # Fixed coordinates avoid RDKit conformer-generation/version variability.
    coords = torch.tensor(
        [[0, 0, 0], [1.43, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89], [1.75, 0.9, 0]],
        dtype=torch.float32,
    ).numpy()
    data = GraphData("methanol", _SMILES["methanol"], confdata={"coords": coords[None]}, max_n_confs=1)
    with torch.no_grad():
        predictions = models["member_01"](data, do_patch=False)
    # kcal/mol and kcal/mol/Angstrom; tolerances intentionally not bitwise strict.
    torch.testing.assert_close(predictions["energy"], torch.tensor([[10.378195]]), rtol=0.002, atol=0.05)
    expected_forces = torch.tensor(
        [
            [14.231960, 7.961950, 0],
            [-2.599747, -22.479368, 0],
            [-4.277336, 3.823990, 0],
            [-4.976168, -4.221879, 5.472404],
            [-4.976141, -4.221885, -5.472411],
            [2.597432, 19.137192, 0],
        ]
    )
    torch.testing.assert_close(predictions["forces"][:, 0], expected_forces, rtol=0.002, atol=0.1)


def test_release_configs_only_contain_inference_model():
    paths = [f"ByteFF-26/models/member-{i:02d}" for i in range(1, 6)] + ["ByteFF-Pol-25/model"]
    for path in paths:
        with open(get_asset_path(f"{path}/fftrainer_config_in_use.yaml")) as file:
            config = yaml.safe_load(file)
        assert set(config) == {"model"}, path
        assert "check_point" not in config["model"], path
