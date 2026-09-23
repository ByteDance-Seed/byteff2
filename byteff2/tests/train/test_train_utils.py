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
import yaml

import byteff2.train.utils as train_utils


class FakeHybridFF:
    def __init__(self, **config):
        self.config = config
        self.loaded_state_dicts = []
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dicts.append((state_dict, strict))

    def eval(self):
        self.eval_called = True
        return self


class CallableModel:
    def __init__(self, ff_layers, forward_result):
        self.ff_block = SimpleNamespace(ff_layers=ff_layers)
        self._forward_result = forward_result
        self.eval_called = False

    def __call__(self, *_args, **_kwargs):
        return self._forward_result

    def parameters(self):
        # get_nb_params calls next(model.parameters()).device; provide a dummy.
        yield torch.zeros(1)

    def eval(self):
        self.eval_called = True
        return self


class _FakeData(dict):
    """Mock GraphData stand-in; supports `.to(device)` and `in` checks."""

    def to(self, _device):
        return self


def test_load_model_from_dict_and_optional_ckpt(monkeypatch, tmp_path):
    monkeypatch.setattr(train_utils, "HybridFF", FakeHybridFF)
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        train_utils.torch,
        "load",
        lambda path, **_kwargs: {"model_state_dict": {"weight": torch.tensor([2.0])}, "path": str(path)},
    )

    model = train_utils.load_model(
        {
            "graph_block": {"feature_layer": {}, "gnn_layer": {}},
            "ff_block": [],
            "supported_elements": [1, 6, 8],
        },
        ckpt=str(ckpt_path),
        device="cpu",
    )

    assert isinstance(model, FakeHybridFF)
    assert model.config == {
        "graph_block": {"feature_layer": {}, "gnn_layer": {}},
        "ff_block": [],
        "supported_elements": [1, 6, 8],
    }
    assert model.eval_called is True
    assert model.loaded_state_dicts[0][0] == {"weight": torch.tensor([2.0])}
    assert model.loaded_state_dicts[0][1] is True


def test_load_model_from_directory_uses_yaml_and_optimal_ckpt(monkeypatch, tmp_path):
    monkeypatch.setattr(train_utils, "HybridFF", FakeHybridFF)

    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    with (model_dir / "fftrainer_config_in_use.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "model": {
                    "graph_block": {"feature_layer": {}, "gnn_layer": {}},
                    "ff_block": [],
                    "supported_elements": [1, 6, 8],
                    "check_point": "to-be-dropped",
                }
            },
            file,
        )

    optimal_path = model_dir / "optimal.pt"
    optimal_path.write_text("placeholder", encoding="utf-8")

    def fake_torch_load(path, **_kwargs):
        assert str(path) == str(optimal_path)
        return {"model_state_dict": {"bias": torch.tensor([1.5])}}

    monkeypatch.setattr(train_utils.torch, "load", fake_torch_load)

    model = train_utils.load_model(str(model_dir), device="cpu", strict=False)

    assert isinstance(model, FakeHybridFF)
    assert model.config == {
        "graph_block": {"feature_layer": {}, "gnn_layer": {}},
        "ff_block": [],
        "supported_elements": [1, 6, 8],
    }
    assert model.device == "cpu"
    assert model.loaded_state_dicts == [({"bias": torch.tensor([1.5])}, False)]
    assert model.eval_called is True


def test_load_model_from_directory_preserves_supported_elements(monkeypatch, tmp_path):
    monkeypatch.setattr(train_utils, "HybridFF", FakeHybridFF)

    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    with (model_dir / "fftrainer_config_in_use.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "model": {
                    "graph_block": {"feature_layer": {}, "gnn_layer": {}},
                    "ff_block": [],
                    "supported_elements": [1, 6, 8, 9, 15, 16],
                }
            },
            file,
        )

    optimal_path = model_dir / "optimal.pt"
    optimal_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(train_utils.torch, "load", lambda *_args, **_kwargs: {"model_state_dict": {}})

    model = train_utils.load_model(str(model_dir), device="cpu")

    assert isinstance(model, FakeHybridFF)
    assert model.config["supported_elements"] == [1, 6, 8, 9, 15, 16]


def test_load_model_requires_supported_elements(monkeypatch):
    monkeypatch.setattr(train_utils, "HybridFF", train_utils.HybridFF)

    with pytest.raises(TypeError, match="supported_elements"):
        train_utils.load_model({"graph_block": {"feature_layer": {}, "gnn_layer": {}}, "ff_block": []})


def test_load_ensemble_model_wraps_loaded_models(monkeypatch):
    loaded = []

    def fake_load_model(path):
        loaded.append(path)
        return f"model:{path}"

    monkeypatch.setattr(train_utils, "load_model", fake_load_model)
    monkeypatch.setattr(train_utils, "EnsembleModel", lambda models: {"models": models})

    ensemble = train_utils.load_ensemble_model(["m1", "m2"])

    assert loaded == ["m1", "m2"]
    assert ensemble == {"models": ["model:m1", "model:m2"]}


def test_load_pretrained_byteff_ensemble_uses_all_members_in_order(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        train_utils,
        "get_asset_path",
        lambda relative, asset_root: f"{asset_root}/{relative}",
    )
    monkeypatch.setattr(
        train_utils,
        "load_model",
        lambda path, device: loaded.append((path, device)) or SimpleNamespace(eval=lambda: None),
    )

    ensemble = train_utils.load_pretrained_model("ByteFF-26", asset_root="/assets", device="cuda")

    assert [path for path, _ in loaded] == [f"/assets/ByteFF-26/models/member-{index:02d}" for index in range(1, 6)]
    assert all(device == "cuda" for _, device in loaded)
    assert len(ensemble.models) == 5


def test_load_pretrained_member_and_byteff_pol(monkeypatch):
    loaded = []
    monkeypatch.setattr(train_utils, "get_asset_path", lambda relative, _root: f"/assets/{relative}")
    monkeypatch.setattr(
        train_utils,
        "load_model",
        lambda path, device: loaded.append((path, device)) or FakeHybridFF(),
    )

    member = train_utils.load_pretrained_model("ByteFF-26", asset_root="/assets", member=3)
    polarizable = train_utils.load_pretrained_model("ByteFF-Pol-25", asset_root="/assets")

    assert member.eval_called is True
    assert polarizable.eval_called is True
    assert loaded == [
        ("/assets/ByteFF-26/models/member-03", "cpu"),
        ("/assets/ByteFF-Pol-25/model", "cpu"),
    ]


@pytest.mark.parametrize("member", [0, 6, True, "1"])
def test_load_pretrained_rejects_invalid_byteff_member(member):
    with pytest.raises(ValueError, match="integer from 1 through 5"):
        train_utils.load_pretrained_model("ByteFF-26", asset_root="/assets", member=member)


def test_load_pretrained_rejects_member_for_byteff_pol():
    with pytest.raises(ValueError, match="does not accept a member"):
        train_utils.load_pretrained_model("ByteFF-Pol-25", asset_root="/assets", member=1)


def test_load_pretrained_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown pretrained model"):
        train_utils.load_pretrained_model("unknown", asset_root="/assets")


def test_load_pretrained_missing_member_does_not_load_partial_ensemble(monkeypatch, tmp_path):
    for index in range(1, 5):
        (tmp_path / "ByteFF-26" / "models" / f"member-{index:02d}").mkdir(parents=True)
    loaded = []
    monkeypatch.setattr(train_utils, "load_model", lambda *args, **kwargs: loaded.append(args))
    with pytest.raises(FileNotFoundError, match="member-05"):
        train_utils.load_pretrained_model("ByteFF-26", asset_root=tmp_path)
    assert not loaded


def test_load_training_config_resolves_paths_and_asset_checkpoint(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "train.yaml"
    config_path.write_text(
        "meta:\n  work_folder: work\ndataset:\n  - config: data/dataset_config.yaml\n"
        "model:\n  check_point_asset: ByteFF-Pol-25/model/optimal.pt\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_utils, "get_asset_path", lambda relative, root: f"{root}/{relative}")

    config = train_utils.load_training_config(str(config_path), asset_root="/assets")

    assert config["meta"]["work_folder"] == str((config_dir / "work").resolve())
    assert config["dataset"][0]["config"] == str((config_dir / "data/dataset_config.yaml").resolve())
    assert config["model"]["check_point"] == "/assets/ByteFF-Pol-25/model/optimal.pt"
    assert "check_point_asset" not in config["model"]


def test_get_nb_params_without_itp_returns_params_and_no_tfs(monkeypatch):
    monkeypatch.setattr(train_utils, "GraphData", lambda *args, **kwargs: _FakeData({"graph": args, "kwargs": kwargs}))
    monkeypatch.setattr(train_utils, "ffparams_to_tfs", lambda *_args, **_kwargs: "unexpected")
    monkeypatch.setattr("byteff2.bytemol.core.rkutil.conformer.find_hypervalent_centers", lambda _rkmol: [])

    ff_parameters = {
        "PreExp6Pol.c6": torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        "PreExp6Pol.rvdw": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        "PreExp6Pol.lambda": torch.tensor([[5.0], [6.0]], dtype=torch.float64),
        "PreExp6Pol.eps": torch.tensor([[7.0], [8.0]], dtype=torch.float64),
        "PreChargeVolume.charges": torch.tensor([[0.2], [-0.2]], dtype=torch.float64),
        "PreExp6Pol.alpha": torch.tensor([[9.0], [10.0]], dtype=torch.float64),
        "PreExp6Pol.pol_damping": torch.tensor([[11.0], [12.0]], dtype=torch.float64),
        "PreExp6Pol.ct_eps": torch.tensor([[13.0], [14.0]], dtype=torch.float64),
        "PreExp6Pol.ct_lamb": torch.tensor([[15.0], [16.0]], dtype=torch.float64),
    }

    class CheckingModel(CallableModel):
        def __call__(self, data, skip_ff=True, do_patch=False):
            assert skip_ff is True
            assert data["graph"][0] == "test"
            return self._forward_result

    fake_model = CheckingModel(
        ff_layers={
            "Exp6Pol": SimpleNamespace(
                s12=12.0,
                disp_damping_factor=0.4,
                dipole_solver=SimpleNamespace(a=0.7),
            )
        },
        forward_result={"ff_parameters": ff_parameters},
    )

    mol = SimpleNamespace(
        name="demo",
        get_mapped_smiles=lambda: "[H:1][O:2][H:3]",
        get_smiles=lambda: "O",
        atomic_numbers=[8, 1, 1],
        conformers=[SimpleNamespace(coords=torch.zeros((3, 3), dtype=torch.float64).numpy())],
        rkmol=object(),
    )

    metadata, params, tfs, returned_mol = train_utils.get_nb_params(fake_model, mol, write_to_itp=False)

    assert metadata["exp6"] is True
    assert metadata["disp_damping"] == pytest.approx(0.4)
    assert metadata["thole"] == pytest.approx(0.7)
    assert params["charge"] == pytest.approx([0.2, -0.2])
    assert params["lamb"] == pytest.approx([5.0, 6.0])
    assert tfs is None
    assert returned_mol is mol
    assert fake_model.eval_called is True
    assert "PreLJEs.sigma" in ff_parameters
    assert "PreLJEs.epsilon" in ff_parameters
    assert "PreLJEs.charges" in ff_parameters


def test_get_nb_params_with_itp_emits_pf6_tfs(monkeypatch):
    monkeypatch.setattr(train_utils, "GraphData", lambda *args, **kwargs: _FakeData())
    monkeypatch.setattr("byteff2.bytemol.core.rkutil.conformer.find_hypervalent_centers", lambda _rkmol: [])
    tfs_stub = object()

    captured = {}

    def fake_ffparams_to_tfs(ffparams, data, mol, mol_name):
        captured["mol_name"] = mol_name
        captured["data"] = data
        captured["charge_shape"] = tuple(ffparams["PreLJEs.charges"].shape)
        return tfs_stub

    monkeypatch.setattr(train_utils, "ffparams_to_tfs", fake_ffparams_to_tfs)

    fake_model = CallableModel(
        ff_layers={
            "Exp6Pol": SimpleNamespace(
                s12=10.0,
                disp_damping_factor=1.2,
                dipole_solver=SimpleNamespace(a=0.39),
            )
        },
        forward_result={
            "ff_parameters": {
                "PreExp6Pol.c6": torch.ones((7, 1), dtype=torch.float64),
                "PreExp6Pol.rvdw": torch.ones((7, 1), dtype=torch.float64),
                "PreExp6Pol.lambda": torch.ones((7, 1), dtype=torch.float64),
                "PreExp6Pol.eps": torch.ones((7, 1), dtype=torch.float64),
                "PreChargeVolume.charges": torch.zeros((7, 1), dtype=torch.float64),
                "PreExp6Pol.alpha": torch.ones((7, 1), dtype=torch.float64),
                "PreExp6Pol.pol_damping": torch.ones((7, 1), dtype=torch.float64),
                "PreExp6Pol.ct_eps": torch.ones((7, 1), dtype=torch.float64),
                "PreExp6Pol.ct_lamb": torch.ones((7, 1), dtype=torch.float64),
            }
        },
    )

    original_coords = torch.zeros((7, 3), dtype=torch.float64).numpy()
    mol = SimpleNamespace(
        name="PF6",
        get_mapped_smiles=lambda: "[P-](F)(F)(F)(F)(F)F",
        get_smiles=lambda: "F[P-](F)(F)(F)(F)F",
        atomic_numbers=[15, 9, 9, 9, 9, 9, 9],
        conformers=[SimpleNamespace(coords=original_coords.copy())],
        rkmol=object(),
    )

    _metadata, _params, tfs, returned_mol = train_utils.get_nb_params(fake_model, mol, write_to_itp=True)

    assert tfs is tfs_stub
    assert returned_mol is mol
    assert captured["mol_name"] == "PF6"
    assert captured["charge_shape"] == (7,)
