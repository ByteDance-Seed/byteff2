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

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml

import byteff2.train.trainer as trainer_mod
from byteff2.train.trainer import FFJointTrainer, FFTrainer, safe_barrier, TrainConfig, TrainState


class FakeGraph(dict):
    def __init__(self, *, batch_size=1, data_idx=None, coords=None):
        super().__init__()
        self["data_idx"] = [0] if data_idx is None else data_idx
        self.counts = torch.zeros((batch_size,), dtype=torch.long)
        self.coords = torch.zeros((batch_size, 3), dtype=torch.float32) if coords is None else coords

    def to(self, _device):
        return self

    def get_count(self, key, idx=None):
        assert key == "node"
        return torch.tensor([self.coords.shape[0]], dtype=torch.long)


class FakeLoader:
    def __init__(self, items, dataset=None):
        self._items = items
        self.dataset = dataset if dataset is not None else SimpleNamespace(data_list=[])

    def __iter__(self):
        return iter(self._items)


class DataRecord(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class FakeScaler:
    def __init__(self):
        self.unscaled = False
        self.stepped = False
        self.updated = False

    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer):
        self.unscaled = True

    def step(self, optimizer):
        self.stepped = True
        optimizer.step()

    def update(self):
        self.updated = True


class GradModeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))
        self.grad_enabled = []
        self.validate_elements = []
        self.train_called = False
        self.eval_called = False

    def train(self, mode=True):
        self.train_called = mode
        return super().train(mode)

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, _graph, *, validate_elements, **_kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.validate_elements.append(validate_elements)
        return {
            "ff_parameters": {
                "MultipoleInt.D_ind": torch.zeros((1, 3), dtype=torch.float32),
                "MultipoleInt.D_ind_cluster": torch.zeros((1, 3), dtype=torch.float32),
            }
        }


class FakeSubsetDataset:
    def __init__(self, data_list, *, rank=0, shard_ids=None):
        self.data_list = list(data_list)
        self.rank = rank
        self.shard_ids = shard_ids if shard_ids is not None else [0]
        self.index_set = False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeSubsetDataset(self.data_list[item], rank=self.rank, shard_ids=self.shard_ids)
        return self.data_list[item]

    def shuffle(self):
        return FakeSubsetDataset(list(reversed(self.data_list)), rank=self.rank, shard_ids=self.shard_ids)

    def set_index(self):
        self.index_set = True


class CapturedLoader:
    def __init__(self, dataset, batch_size, shuffle, drop_last, collate_fn, num_workers=0, sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.sampler = sampler

    def __iter__(self):
        return iter([])


class FFOptGraph:
    def __init__(self):
        self.coords = torch.tensor(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        self.confmask = torch.ones((2, 2), dtype=torch.float32)
        self.inc_node_torsion_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        self.mol_name = ["mol_a", "mol_b"]

    def to(self, _device):
        return self

    def get_count(self, key, idx=None):
        assert key == "node"
        return torch.tensor([2, 2], dtype=torch.long)


def _config_dict(work_folder: Path) -> dict:
    return {
        "meta": {"work_folder": str(work_folder), "random_seed": 7, "fp64": True},
        "dataset": [{"config": "dataset.yaml", "batch_size": 4}],
        "model": {"supported_elements": [1, 6, 8]},
        "training": {"optimizer": {"type": "SGD", "lr": 0.1}},
    }


def test_train_config_merges_defaults_and_writes_yaml(tmp_path):
    cfg = TrainConfig(_config_dict(tmp_path / "run"), timestamp=False, make_working_dir=True)

    assert cfg.dataset[0]["config"] == "dataset.yaml"
    assert cfg.dataset[0]["batch_size"] == 4
    assert cfg.dataset[0]["train_ratio"] == pytest.approx(0.9)
    assert cfg.dataset[0]["shuffle"] is True
    assert cfg.meta["random_seed"] == 7
    assert Path(cfg.ckpt_folder).is_dir()
    assert Path(cfg.work_folder, "fftrainer_config_in_use.yaml").is_file()


def test_train_config_existing_workdir_error_is_explicit(tmp_path):
    work_folder = tmp_path / "existing_run"
    work_folder.mkdir()

    with pytest.raises(AssertionError, match=r"already exists\."):
        TrainConfig(_config_dict(work_folder), timestamp=False, make_working_dir=True)


def test_train_config_from_yaml_and_state_transitions(tmp_path):
    config_path = tmp_path / "config.yaml"
    work_folder = tmp_path / "yaml_run"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_config_dict(work_folder), file)

    cfg = TrainConfig(str(config_path), timestamp=False, make_working_dir=True)
    assert cfg.train_state == TrainState.NULL
    assert cfg.finish_flag == str(Path(cfg.work_folder) / "FINISHED")
    assert cfg.optimal_path() == str(Path(cfg.work_folder) / "optimal.pt")
    assert cfg.optimal_path("best") == str(Path(cfg.work_folder) / "optimal_best.pt")

    torch.save({"model_state_dict": {}}, cfg.optimal_path())
    assert cfg.train_state == TrainState.STARTED

    Path(cfg.finish_flag).touch()
    assert cfg.train_state == TrainState.FINISHED


def test_get_latest_ckpt_picks_highest_epoch(tmp_path):
    cfg = TrainConfig(_config_dict(tmp_path / "latest_ckpt"), timestamp=False, make_working_dir=True)
    Path(cfg.ckpt_folder, "ckpt_epoch_2.pt").touch()
    Path(cfg.ckpt_folder, "ckpt_epoch_10.pt").touch()
    Path(cfg.ckpt_folder, "ignore.txt").touch()

    assert cfg.get_latest_ckpt() == str(Path(cfg.ckpt_folder) / "ckpt_epoch_10.pt")


def test_safe_barrier_only_calls_dist_when_initialized(monkeypatch):
    barrier = MagicMock(return_value="barrier-called")
    monkeypatch.setattr(trainer_mod.dist, "barrier", barrier)
    monkeypatch.setattr(trainer_mod.dist, "is_initialized", lambda: False)
    assert safe_barrier() is None
    barrier.assert_not_called()

    monkeypatch.setattr(trainer_mod.dist, "is_initialized", lambda: True)
    assert safe_barrier() == "barrier-called"
    barrier.assert_called_once_with()


def test_init_optimizer_scheduler_supports_grouped_lr_and_scheduler():
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.logger = MagicMock()
    trainer.config = SimpleNamespace(
        training={
            "optimizer": {"type": "Adam", "lr": {"bond": 1e-3, "angle": 2e-3}, "weight_decay": 0.01},
            "scheduler": {"type": "StepLR", "step_size": 3, "gamma": 0.5},
        }
    )
    params = {
        "bond": torch.nn.Parameter(torch.tensor([1.0])),
        "angle": torch.nn.Parameter(torch.tensor([2.0])),
    }
    trainer.model = SimpleNamespace(get_parameters=lambda name: [params[name]])

    trainer._init_optimizer_scheduler()

    assert sorted(group["lr"] for group in trainer.optimizer.param_groups) == [1e-3, 2e-3]
    assert trainer.optimizer.defaults["weight_decay"] == pytest.approx(0.01)
    assert trainer.scheduler is not None
    assert trainer.scheduler.step_size == 3
    assert trainer.scheduler.gamma == pytest.approx(0.5)


def test_save_load_ckpt_and_optimal_roundtrip(tmp_path):
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.logger = MagicMock()
    ckpt_folder = tmp_path / "ckpt"
    ckpt_folder.mkdir()
    optimal_file = tmp_path / "optimal.pt"
    trainer.config = SimpleNamespace(
        ckpt_folder=str(ckpt_folder),
        optimal_path=lambda label="": str(optimal_file if not label else tmp_path / f"optimal_{label}.pt"),
    )
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=2, gamma=0.5)
    trainer.optimal_state_dict = None
    trainer.epoch = 3
    trainer.ffopt_iter = 1
    trainer.ffopt_begin_epoch = [0, 2]
    trainer.best_valid_loss = 0.25
    trainer.early_stop_count = 2
    trainer.train_history = [[("train", 1.0)]]
    trainer.valid_history = [[("valid", 0.5)]]
    trainer.aux_history = [[("aux", 0.2)]]
    trainer.trainer_state_variables = [
        "epoch",
        "ffopt_iter",
        "ffopt_begin_epoch",
        "best_valid_loss",
        "early_stop_count",
        "train_history",
        "valid_history",
        "aux_history",
    ]

    trainer.save_ckpt()
    ckpt_path = ckpt_folder / "ckpt_epoch_3.pt"
    assert ckpt_path.is_file()

    saved_weight = trainer.model.weight.detach().clone()
    saved_bias = trainer.model.bias.detach().clone()

    with torch.no_grad():
        trainer.model.weight.add_(10.0)
        trainer.model.bias.add_(10.0)
    trainer.epoch = 99
    trainer.ffopt_iter = 99
    trainer.best_valid_loss = 99.0
    trainer.train_history = []
    trainer.valid_history = []
    trainer.aux_history = []

    trainer.load_ckpt(str(ckpt_path), model_only=False)

    assert torch.allclose(trainer.model.weight, saved_weight)
    assert torch.allclose(trainer.model.bias, saved_bias)
    assert trainer.epoch == 3
    assert trainer.ffopt_iter == 1
    assert trainer.best_valid_loss == pytest.approx(0.25)
    assert trainer.train_history == [[("train", 1.0)]]
    assert trainer.valid_history == [[("valid", 0.5)]]
    assert trainer.aux_history == [[("aux", 0.2)]]

    trainer.save_optimal()
    assert optimal_file.is_file()
    optimal_weight = trainer.model.weight.detach().clone()

    with torch.no_grad():
        trainer.model.weight.zero_()
    trainer.load_optimal()

    assert torch.allclose(trainer.model.weight, optimal_weight)


def test_joint_trainer_ckpt_roundtrip_preserves_ffopt_state(tmp_path):
    trainer = FFJointTrainer.__new__(FFJointTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.logger = MagicMock()
    ckpt_folder = tmp_path / "ckpt_joint"
    ckpt_folder.mkdir()
    trainer.config = SimpleNamespace(
        ckpt_folder=str(ckpt_folder),
        optimal_path=lambda label="": str(tmp_path / "joint_optimal.pt"),
    )
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=2, gamma=0.5)
    trainer.optimal_state_dict = None
    trainer.epoch = 3
    trainer.ffopt_iter = 1
    trainer.ffopt_begin_epoch = [0, 2]
    trainer.best_valid_loss = 0.25
    trainer.early_stop_count = 2
    trainer.train_history = [[("train", 1.0)]]
    trainer.valid_history = [[("valid", 0.5)]]
    trainer.aux_history = [[("aux", 0.2)]]
    trainer.trainer_state_variables = [
        "epoch",
        "best_valid_loss",
        "early_stop_count",
        "train_history",
        "valid_history",
        "aux_history",
        "ffopt_iter",
        "ffopt_begin_epoch",
    ]

    trainer.save_ckpt()

    trainer.epoch = 99
    trainer.ffopt_iter = 99
    trainer.ffopt_begin_epoch = []
    trainer.best_valid_loss = 99.0
    trainer.train_history = []
    trainer.valid_history = []
    trainer.aux_history = []

    trainer.load_ckpt(str(ckpt_folder / "ckpt_epoch_3.pt"), model_only=False)

    assert trainer.epoch == 3
    assert trainer.ffopt_iter == 1
    assert trainer.ffopt_begin_epoch == [0, 2]
    assert trainer.best_valid_loss == pytest.approx(0.25)
    assert trainer.train_history == [[("train", 1.0)]]


def test_valid_epoch_aggregates_losses_records_histories_and_toggles_grad(monkeypatch):
    monkeypatch.setattr(trainer_mod, "safe_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "convert_conj_to_bonded_params", lambda _ffparams: None)

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.epoch = 3
    trainer.logger = MagicMock()
    trainer.model = GradModeModel()
    trainer.config = SimpleNamespace(
        dataset=[
            {
                "loss_weight": 1.0,
                "loss": [{"loss_type": "ParamMSE", "weight": 1.0}],
                "aux_loss": [{"loss_type": "L1_Norm"}],
            },
            {
                "loss_weight": 0.5,
                "cluster": True,
                "loss": [{"loss_type": "ParamMSE", "weight": 1.0}],
                "aux_loss": [{"loss_type": "L1_Norm"}],
            },
        ]
    )
    trainer.valid_history = [[], []]
    trainer.aux_history = [[], []]
    trainer.valid_dls = [FakeLoader([FakeGraph(batch_size=2)]), FakeLoader([FakeGraph(batch_size=1)])]

    def fake_calc_loss(_pred, _graph, dataset_index, is_valid=True):
        assert is_valid is True
        values = {0: [torch.tensor(2.0), torch.tensor(0.2)], 1: [torch.tensor(4.0), torch.tensor(0.4)]}
        return values[dataset_index], []

    trainer.calc_loss = fake_calc_loss

    averaged = trainer.valid_epoch()

    assert averaged.item() == pytest.approx(4.0)
    assert trainer.model.eval_called is True
    assert trainer.model.grad_enabled == [False, False]
    assert trainer.model.validate_elements == [False, False]
    assert trainer.valid_history == [[[3, 0, 2.0]], [[3, 0, 4.0]]]
    assert trainer.aux_history[0][0][:2] == [3, 0]
    assert trainer.aux_history[1][0][:2] == [3, 0]
    assert trainer.aux_history[0][0][2] == pytest.approx(0.2)
    assert trainer.aux_history[1][0][2] == pytest.approx(0.4)


def test_valid_epoch_enables_grad_for_inter_energy_losses(monkeypatch):
    monkeypatch.setattr(trainer_mod, "safe_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "convert_conj_to_bonded_params", lambda _ffparams: None)

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.epoch = 1
    trainer.logger = MagicMock()
    trainer.model = GradModeModel()
    trainer.config = SimpleNamespace(
        dataset=[
            {
                "loss_weight": 1.0,
                "cluster": True,
                "loss": [{"loss_type": "InterEnergyMSE", "weight": 1.0}],
            }
        ]
    )
    trainer.valid_history = [[]]
    trainer.aux_history = [[]]
    trainer.valid_dls = [FakeLoader([FakeGraph(batch_size=1)])]
    trainer.calc_loss = lambda _pred, _graph, _dataset_index, is_valid=True: ([torch.tensor(3.0)], [])

    averaged = trainer.valid_epoch()

    assert averaged.item() == pytest.approx(3.0)
    assert trainer.model.grad_enabled == [True]
    assert trainer.model.validate_elements == [False]
    assert trainer.valid_history == [[[1, 0, 3.0]]]


def test_train_epoch_handles_nan_loss_and_grad_clip(monkeypatch):
    monkeypatch.setattr(trainer_mod, "safe_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "convert_conj_to_bonded_params", lambda _ffparams: None)

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.epoch = 1
    trainer.epoch_step_num = 1
    trainer.logger = MagicMock()
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer.model = GradModeModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.amp_scaler = FakeScaler()
    trainer.train_history = [[]]
    dataset_item = DataRecord(data_idx=0, coords=torch.zeros((1, 3), dtype=torch.float32))
    trainer.train_dls = [
        FakeLoader([FakeGraph(batch_size=1, data_idx=[0])], dataset=SimpleNamespace(data_list=[dataset_item]))
    ]
    trainer.config = SimpleNamespace(
        dataset=[{"loss_weight": 2.0}],
        training={"grad_clip": 0.5},
    )

    def fake_calc_loss(_pred, _graph, dataset_index, is_valid=False):
        assert dataset_index == 0
        assert is_valid is False
        return [trainer.model.param * 0 + torch.tensor(float("nan"))], [float("nan")]

    trainer.calc_loss = fake_calc_loss

    trainer.train_epoch()

    assert trainer.model.train_called is True
    assert trainer.amp_scaler.unscaled is True
    assert trainer.amp_scaler.stepped is True
    assert trainer.amp_scaler.updated is True
    assert trainer.model.validate_elements == [False]
    assert len(trainer.train_history) == 1
    assert len(trainer.train_history[0]) == 1
    assert trainer.train_history[0][0][:2] == [1, 0]
    assert math.isnan(trainer.train_history[0][0][2])
    assert "D_ind" not in dataset_item
    assert "D_ind_cluster" not in dataset_item
    trainer.logger.warning.assert_called()


def test_train_and_valid_drives_scheduler_ckpt_optimal_and_plotting():
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.logger = MagicMock()
    trainer.epoch = 0
    trainer.best_valid_loss = float("inf")
    trainer.early_stop_count = 0
    trainer.config = SimpleNamespace(
        training={
            "ckpt_interval": 1,
            "max_epoch": 2,
            "valid_interval": 1,
            "ignore_tolerance": 0.0,
            "early_stop_patience": 10,
        }
    )
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 0.01}])
    scheduler_calls = []

    def scheduler_step(loss):
        scheduler_calls.append(loss)

    trainer.scheduler = SimpleNamespace(step=scheduler_step)
    ckpts = []
    optimals = []
    plots = []
    trains = []
    valid_values = iter([5.0, 4.0])
    trainer.save_ckpt = lambda: ckpts.append(trainer.epoch)
    trainer.valid_epoch = lambda: next(valid_values)
    trainer.save_optimal = lambda: optimals.append(trainer.epoch)
    trainer.plot_history = lambda: plots.append(trainer.epoch)
    trainer.train_epoch = lambda: trains.append(trainer.epoch)

    final_epoch = trainer.train_and_valid()

    assert final_epoch == 2
    assert scheduler_calls == [5.0, 4.0]
    assert ckpts == [0, 1, 2]
    assert optimals == [0, 1]
    assert trains == [0, 1]
    assert plots == [0, 0, 1, 1]
    assert trainer.best_valid_loss == pytest.approx(4.0)
    assert trainer.early_stop_count == 0


def test_train_loop_runs_single_stage_for_base_trainer(tmp_path):
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.logger = MagicMock()
    trainer.restarted = True
    trainer.config = SimpleNamespace(
        work_folder=str(tmp_path),
        finish_flag=str(Path(tmp_path) / "FINISHED"),
    )

    init_calls = []

    def init_optimizer_scheduler():
        init_calls.append("init")

    def train_and_valid():
        init_calls.append("train_and_valid")

    trainer._init_optimizer_scheduler = init_optimizer_scheduler
    trainer.train_and_valid = train_and_valid

    trainer.train_loop()

    assert init_calls == ["train_and_valid"]
    assert Path(trainer.config.work_folder, "FINISHED").is_file()


def test_joint_train_loop_skips_reinit_once_when_restarted_and_marks_finished(tmp_path):
    trainer = FFJointTrainer.__new__(FFJointTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.logger = MagicMock()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.restarted = True
    trainer.ffopt_iter = 0
    trainer.epoch = 7
    trainer.ffopt_begin_epoch = [0]
    trainer.best_valid_loss = 123.0
    trainer.early_stop_count = 5
    trainer.config = SimpleNamespace(
        training={"max_ffopt_iters": 2},
        work_folder=str(tmp_path),
        finish_flag=str(Path(tmp_path) / "FINISHED"),
    )

    init_calls = []
    ffopt_calls = []
    train_valid_calls = []
    load_optimal_calls = []
    ckpt_calls = []

    def init_optimizer_scheduler():
        init_calls.append(trainer.ffopt_iter)

    def ffopt_all(iter_idx):
        ffopt_calls.append((iter_idx, trainer.model.training))

    def train_and_valid():
        train_valid_calls.append((trainer.ffopt_iter, trainer.model.training))

    def load_optimal():
        load_optimal_calls.append(trainer.ffopt_iter)

    def save_ckpt():
        ckpt_calls.append(trainer.ffopt_iter)

    trainer._init_optimizer_scheduler = init_optimizer_scheduler
    trainer.ffopt_all = ffopt_all
    trainer.train_and_valid = train_and_valid
    trainer.load_optimal = load_optimal
    trainer.save_ckpt = save_ckpt

    trainer.train_loop()

    assert init_calls == [1]
    assert ffopt_calls == [(0, False), (1, False)]
    assert train_valid_calls == [(0, True), (1, True)]
    assert load_optimal_calls == [0, 1]
    assert ckpt_calls == [1, 2]
    assert trainer.ffopt_iter == 2
    assert trainer.ffopt_begin_epoch == [0, 7, 7]
    assert trainer.early_stop_count == 0
    assert trainer.best_valid_loss == torch.finfo(torch.get_default_dtype()).max
    assert Path(trainer.config.work_folder, "FINISHED").is_file()


def test_plot_history_writes_json_and_images(tmp_path):
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.config = SimpleNamespace(
        work_folder=str(tmp_path),
        dataset=[
            {
                "loss": [
                    {"loss_type": "ParamMSE", "weight": 1.0},
                    {"loss_type": "BondedEnergy", "weight": 0.5},
                ],
                "aux_loss": [{"loss_type": "L1_Norm"}],
            }
        ],
    )
    trainer.train_dls = [object()]
    trainer.train_history = [[[0, 0, 1.0], [0, 1, 100.0], [1, 0, 2.0]]]
    trainer.valid_history = [[[0, 0, 4.0], [1, 0, 3.0]]]
    trainer.aux_history = [[[0, 0, 0.5], [1, 0, 0.25]]]
    trainer.ffopt_begin_epoch = [0, 1]

    trainer.plot_history()

    assert Path(tmp_path, "history.json").is_file()
    assert Path(tmp_path, "history.jpg").is_file()
    assert Path(tmp_path, "aux_history_0.jpg").is_file()
    assert Path(tmp_path, "history.json").read_text(encoding="utf-8").startswith("{\n  ")


def test_init_logger_writes_log_path_and_commit(monkeypatch, tmp_path):
    logger = MagicMock()
    monkeypatch.setattr(trainer_mod, "setup_default_logging", lambda stdout, file_path: logger)
    monkeypatch.setattr(
        trainer_mod,
        "Repo",
        lambda **_kwargs: SimpleNamespace(head=SimpleNamespace(commit=SimpleNamespace(hexsha="abc123"))),
    )

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.write_log_file = True
    trainer.config = SimpleNamespace(work_folder=str(tmp_path))

    got = trainer._init_logger()

    assert got is logger
    logger.info.assert_any_call(f"writing logs to {tmp_path / 'fftrainer.log'}")
    logger.info.assert_any_call("current commit: abc123")


def test_init_logger_skips_rank_zero_messages_when_local_rank_is_nonzero(monkeypatch, tmp_path):
    logger = MagicMock()
    monkeypatch.setattr(trainer_mod, "setup_default_logging", lambda stdout, file_path: logger)
    repo = MagicMock()
    monkeypatch.setattr(trainer_mod, "Repo", repo)

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 1
    trainer.write_log_file = True
    trainer.config = SimpleNamespace(work_folder=str(tmp_path))

    got = trainer._init_logger()

    assert got is logger
    logger.info.assert_not_called()
    repo.assert_not_called()


def test_init_constructs_state_and_calls_start_ddp(monkeypatch):
    fake_config = SimpleNamespace(dataset=[{"demo": 1}], meta={"fp64": False}, training={}, work_folder="workdir")
    start_calls = []
    monkeypatch.setattr(trainer_mod, "TrainConfig", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(FFTrainer, "_init_logger", lambda self: MagicMock())
    monkeypatch.setattr(
        FFTrainer,
        "start_ddp",
        lambda self, rank, world_size, device, restart=False, load_ckpt=True: start_calls.append(
            (rank, world_size, str(device), restart, load_ckpt)
        ),
    )

    trainer = FFTrainer(config={"x": 1}, timestamp=False, ddp=False, device="cpu", load_data=False, use_amp=True)

    assert trainer.rank == 0
    assert trainer.world_size == 1
    assert not hasattr(trainer, "ffopt_begin_epoch")
    assert trainer.train_history == [[]]
    assert trainer.valid_history == [[]]
    assert trainer.aux_history == [[]]
    assert trainer.use_amp is True
    assert start_calls == [(0, 1, "cpu", False, True)]


def test_joint_trainer_init_constructs_ffopt_state(monkeypatch):
    fake_config = SimpleNamespace(dataset=[{"demo": 1}], meta={"fp64": False}, training={}, work_folder="workdir")
    start_calls = []
    monkeypatch.setattr(trainer_mod, "TrainConfig", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(FFJointTrainer, "_init_logger", lambda self: MagicMock())
    monkeypatch.setattr(
        FFJointTrainer,
        "start_ddp",
        lambda self, rank, world_size, device, restart=False, load_ckpt=True: start_calls.append(
            (rank, world_size, str(device), restart, load_ckpt)
        ),
    )

    trainer = FFJointTrainer(config={"x": 1}, timestamp=False, ddp=False, restart=False, use_amp=True)

    assert trainer.ffopt_iter == 0
    assert trainer.ffopt_begin_epoch == [0]
    assert trainer.train_history == [[]]
    assert trainer.valid_history == [[]]
    assert trainer.aux_history == [[]]
    assert trainer.use_amp is True
    assert start_calls == [(0, 1, "cuda:0", False, True)]


def test_start_ddp_loads_data_wraps_ddp_and_restarts(monkeypatch):
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.load_data = True
    trainer.restarted = False
    trainer.epoch = 5
    events = []

    def init_logger():
        return MagicMock()

    trainer._init_logger = init_logger
    seed_calls = []

    def set_seed(seed):
        seed_calls.append(seed)

    trainer._set_seed = set_seed
    trainer._load_data = lambda cfg: events.append("load_data") or (["dataset"], ["train_dl"], ["valid_dl"])
    init_calls = []
    load_calls = []
    trainer._init_optimizer_scheduler = lambda: init_calls.append("init")
    trainer.load_ckpt = lambda path, model_only=True: (
        events.append("load_ckpt"),
        load_calls.append((path, model_only)),
    )
    monkeypatch.setattr(trainer_mod, "safe_barrier", lambda: init_calls.append("barrier"))

    class FakeHybrid:
        def __init__(self, **config):
            events.append("init_model")
            self.config = config
            self.device = None

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(trainer_mod, "HybridFF", FakeHybrid)
    monkeypatch.setattr(trainer_mod, "DDP", lambda model, find_unused_parameters=False: SimpleNamespace(module=model))

    trainer.config = SimpleNamespace(
        meta={"fp64": False, "random_seed": 17},
        dataset=[{"config": "ds.yaml"}],
        model={"supported_elements": [1, 6, 8], "check_point": "warmup.pt"},
        get_latest_ckpt=lambda: "latest.pt",
    )

    trainer.start_ddp(rank=0, world_size=2, device="cpu", find_unused_parameters=True, restart=True, load_ckpt=True)

    assert trainer.datasets == ["dataset"]
    assert trainer.train_dls == ["train_dl"]
    assert trainer.valid_dls == ["valid_dl"]
    assert seed_calls == [17]
    assert init_calls == ["init", "barrier"]
    assert load_calls == [("latest.pt", False)]
    assert trainer.restarted is True
    assert trainer.world_size == 2
    assert trainer.local_rank == 0
    assert isinstance(trainer.model.module, FakeHybrid)
    assert trainer.model.module.device == torch.device("cpu")
    assert events == ["init_model", "load_ckpt", "load_data"]
    assert trainer.config.model == {"supported_elements": [1, 6, 8]}


def test_start_ddp_loads_checkpoint_when_not_restarted(monkeypatch):
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.load_data = False
    trainer.restarted = False

    def init_logger():
        return MagicMock()

    def set_seed(_seed):
        return None

    trainer._init_logger = init_logger
    trainer._set_seed = set_seed
    trainer.load_ckpt = MagicMock()
    monkeypatch.setattr(
        trainer_mod, "HybridFF", lambda **_kwargs: SimpleNamespace(to=lambda _device: SimpleNamespace())
    )
    trainer.config = SimpleNamespace(
        meta={"fp64": True, "random_seed": 3},
        dataset=[],
        model={"check_point": "initial.pt"},
    )

    trainer.start_ddp(rank=0, world_size=1, device="cpu", restart=False, load_ckpt=True)

    trainer.load_ckpt.assert_called_once_with("initial.pt", model_only=True)


def test_start_ddp_uses_explicit_local_rank(monkeypatch):
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.load_data = False
    trainer.restarted = False

    def init_logger():
        return MagicMock()

    trainer._init_logger = init_logger
    trainer._set_seed = lambda _seed: None
    trainer.load_ckpt = MagicMock()
    monkeypatch.setattr(
        trainer_mod, "HybridFF", lambda **_kwargs: SimpleNamespace(to=lambda _device: SimpleNamespace())
    )
    monkeypatch.setattr(trainer_mod, "DDP", lambda model, find_unused_parameters=False: SimpleNamespace(module=model))
    trainer.config = SimpleNamespace(
        meta={"fp64": True, "random_seed": 3},
        dataset=[],
        model={},
    )

    trainer.start_ddp(rank=3, world_size=4, device="cpu", restart=False, load_ckpt=False, local_rank=1)

    assert trainer.rank == 3
    assert trainer.local_rank == 1


def test_train_valid_split_respects_data_num_and_shuffle(monkeypatch):
    monkeypatch.setattr(trainer_mod, "DataLoader", CapturedLoader)

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.logger = MagicMock()
    seed_calls = []

    def set_seed(seed):
        seed_calls.append(seed)

    trainer._set_seed = set_seed
    trainer.config = SimpleNamespace(meta={"dataset_seed": 11, "random_seed": 7})
    dataset = FakeSubsetDataset(
        [
            DataRecord(name="AA_demo", coords=torch.zeros((2, 3))),
            DataRecord(name="BB_demo", coords=torch.ones((2, 3))),
            DataRecord(name="CC_demo", coords=torch.full((2, 3), 2.0)),
            DataRecord(name="AA_tail", coords=torch.full((2, 3), 3.0)),
        ]
    )

    train_dl, valid_dl, steps = trainer._train_valid_split(
        dataset,
        {
            "data_num": 3,
            "shuffle": True,
            "train_ratio": 0.5,
            "batch_size": 5,
        },
    )

    assert train_dl.batch_size == 2
    assert train_dl.shuffle is True
    assert train_dl.drop_last is False
    assert valid_dl.batch_size == 5
    assert steps == 1
    assert train_dl.sampler is None
    assert valid_dl.sampler is None
    assert seed_calls == [11, 7]
    assert train_dl.dataset.index_set is False
    assert valid_dl.dataset.index_set is False
    assert "D_ind" not in train_dl.dataset.data_list[0]
    assert "D_ind_cluster" not in valid_dl.dataset.data_list[0]


def test_load_data_handles_finetune_and_regular_paths(monkeypatch):
    monkeypatch.setattr(trainer_mod, "DataLoader", CapturedLoader)
    created = []

    def fake_imdataset(config, rank, world_size, shard_id=None):
        ds = FakeSubsetDataset(
            [DataRecord(name=f"{config}_0", coords=torch.zeros((1, 3))) for _ in range(4)], rank=rank
        )
        ds.config = config
        ds.shard_ids = [shard_id]
        created.append((config, rank, world_size, shard_id))
        return ds

    monkeypatch.setattr(trainer_mod, "IMDataset", fake_imdataset)
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.rank = 1
    trainer.local_rank = 1
    trainer.world_size = 2
    trainer.device = torch.device("cpu")
    trainer.load_shards = [9, 8]
    trainer.logger = MagicMock()
    trainer._train_valid_split = lambda dataset, config: (f"train:{dataset.config}", f"valid:{dataset.config}", 3)

    reduce_calls = []
    monkeypatch.setattr(
        trainer_mod.dist, "all_reduce", lambda tensor, op=None: (reduce_calls.append(tensor.item()), tensor.fill_(2))
    )

    datasets, train_dls, valid_dls = trainer._load_data(
        [
            {"config": "train_cfg", "valid_root": "v", "valid_config": "valid_cfg", "batch_size": 2, "shuffle": True},
            {"config": "regular_cfg", "batch_size": 2},
        ]
    )

    assert len(datasets) == 2
    assert train_dls[0].batch_size == 2
    assert valid_dls[0].batch_size == 2
    assert train_dls[1] == "train:regular_cfg"
    assert valid_dls[1] == "valid:regular_cfg"
    assert trainer.epoch_step_num == 2
    assert created == [
        ("train_cfg", 1, 2, None),
        ("valid_cfg", 1, 2, None),
        ("regular_cfg", 1, 2, 8),
    ]
    assert reduce_calls == [2]


def test_load_ckpt_falls_back_to_strict_false_on_shape_mismatch(monkeypatch):
    class FlakyModel:
        def __init__(self):
            self.calls = []

        def load_state_dict(self, state_dict, strict=True):
            self.calls.append((dict(state_dict), strict))
            if strict:
                raise RuntimeError("shape mismatch")

        def state_dict(self):
            return {"good": torch.zeros((1,)), "bad": torch.zeros((2,))}

    trainer = FFTrainer.__new__(FFTrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 1
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.logger = MagicMock()
    trainer.model = FlakyModel()
    monkeypatch.setattr(
        trainer_mod.torch,
        "load",
        lambda *_args, **_kwargs: {"model_state_dict": {"good": torch.ones((1,)), "bad": torch.ones((3,))}},
    )

    trainer.load_ckpt("dummy.pt", model_only=True)

    assert len(trainer.model.calls) == 2
    assert trainer.model.calls[1] == ({"good": torch.ones((1,))}, False)


def test_calc_loss_applies_weights_and_aux_only_for_validation(monkeypatch):
    calls = []

    def fake_loss_func(_pred, _graph, loss_type, **kwargs):
        calls.append((loss_type.name, kwargs))
        values = {"ParamMSE": torch.tensor(2.0), "L1_Norm": torch.tensor(5.0)}
        return values[loss_type.name]

    monkeypatch.setattr(trainer_mod, "loss_func", fake_loss_func)
    trainer = FFTrainer.__new__(FFTrainer)
    trainer.config = SimpleNamespace(
        dataset=[
            {
                "loss": [{"loss_type": "ParamMSE", "weight": 3.0, "valid_weight": 4.0, "kwargs": {"x": 1}}],
                "aux_loss": [{"loss_type": "L1_Norm", "kwargs": {"y": 2}}],
            }
        ]
    )

    valid_losses, valid_sep_losses = trainer.calc_loss({}, {}, 0, is_valid=True)
    train_losses, train_sep_losses = trainer.calc_loss({}, {}, 0, is_valid=False)

    assert valid_losses[0].item() == pytest.approx(8.0)
    assert valid_losses[1].item() == pytest.approx(5.0)
    assert valid_sep_losses == [pytest.approx(2.0)]
    assert train_losses[0].item() == pytest.approx(6.0)
    assert train_sep_losses == [pytest.approx(2.0)]
    assert calls == [
        ("ParamMSE", {"x": 1}),
        ("L1_Norm", {"y": 2}),
        ("ParamMSE", {"x": 1}),
    ]


def test_ffopt_replaces_nan_and_nonconverged_coordinates(monkeypatch):
    graph = FFOptGraph()
    energy_force_calls = []
    jacobian_calls = []
    optimize_kwargs = {}

    def fake_energy_force(_graph, _ff_params, calc_partial_hessian=False, confmask=None):
        energy_force_calls.append((calc_partial_hessian, confmask.clone()))
        return {"total_energy": torch.tensor([1.0]), "total_forces": torch.tensor([2.0])}

    def fake_jacobian(coords, torsion_ids):
        jacobian_calls.append((coords.clone(), torsion_ids.clone()))
        return torch.tensor([0.1]), torch.tensor([0.2])

    def fake_optimize(_graph, energy_func, jacobian_func, **config):
        optimize_kwargs.update(config)
        energy_func(_graph.coords)
        jacobian_func(_graph.coords)
        return (
            torch.tensor(
                [
                    [[float("nan"), 8.0, 8.0], [8.0, 8.0, 8.0]],
                    [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([[False, False], [True, True]]),
            4,
        )

    monkeypatch.setattr(trainer_mod, "dihedral_jacobian", fake_jacobian)
    monkeypatch.setattr(trainer_mod.ConstraintFFopt, "optimize", fake_optimize)
    monkeypatch.setattr(trainer_mod, "batch_to_atoms", lambda mask, _counts: mask.unsqueeze(-1).expand(-1, -1, 3))

    trainer = FFJointTrainer.__new__(FFJointTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.ffopt_iter = 1
    trainer.logger = MagicMock()

    class FakeModel:
        def __call__(self, _graph, skip_ff=False, validate_elements=True):
            return {"energy": torch.tensor([1.0]), "forces": torch.tensor([2.0])}

    trainer.model = FakeModel()

    result = trainer.ffopt(graph, {"pos_res_k": [0.5, 1.5], "max_iter": 3})

    assert optimize_kwargs["pos_res_k"] == 1.5
    assert optimize_kwargs["max_iter"] == 3
    assert len(jacobian_calls) == 1
    assert torch.allclose(result[0], graph.coords[0])
    assert torch.allclose(result[1], torch.full((2, 3), 9.0, dtype=torch.float32))


def test_ffopt_all_runs_ffopt_pipeline_and_rebuilds_loaders(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer_mod, "DataLoader", lambda dataset, **kwargs: [FakeGraph(batch_size=1)])
    monkeypatch.setattr(trainer_mod.torch.cuda, "empty_cache", lambda: None)

    class FakeFFOptDataset:
        processed_names = ["proc.bin"]

        def __init__(self):
            self.saved = None
            self.updated = []
            self.load_calls = []

        def load(self, *args):
            self.load_calls.append(args)
            return 0 if args else None

        def update_data(self, coords, counts, offset, key):
            self.updated.append((coords.clone(), counts.clone(), offset, key))

        def save(self, save_dir=None, label=None):
            self.saved = (save_dir, label)

    dataset = FakeFFOptDataset()
    trainer = FFJointTrainer.__new__(FFJointTrainer)
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.logger = MagicMock()
    trainer.ffopt = lambda graph, cfg: torch.full((1, 1, 3), 7.0)
    trainer._train_valid_split = lambda _dataset, _config: ("new_train", "new_valid", 2)
    trainer.config = SimpleNamespace(work_folder=str(tmp_path), dataset=[{"ffopt": {"k": 1}, "ffopt_batch_size": 8}])
    trainer.datasets = [dataset]
    trainer.train_dls = ["old_train"]
    trainer.valid_dls = ["old_valid"]

    trainer.ffopt_all(3)

    assert dataset.load_calls == [(str(tmp_path), "_ffopt_3"), ()]
    assert dataset.updated[0][3] == "coords"
    assert dataset.saved == (str(tmp_path), "_ffopt_3")
    assert trainer.train_dls == ["new_train"]
    assert trainer.valid_dls == ["new_valid"]


def test_ffjoint_ffopt_uses_model_energy_and_replaces_nonconverged(monkeypatch):
    graph = FFOptGraph()
    optimize_calls = []
    model_calls = []

    def fake_optimize(_graph, energy_func, jacobian_func, **config):
        optimize_calls.append(config)
        energy, forces = energy_func(_graph.coords)
        assert torch.allclose(energy, torch.tensor([3.0]))
        assert torch.allclose(forces, torch.tensor([4.0]))
        jacobian_func(_graph.coords)
        return torch.full_like(_graph.coords, 6.0), torch.tensor([[True, True], [False, False]]), 2

    monkeypatch.setattr(trainer_mod.ConstraintFFopt, "optimize", fake_optimize)
    monkeypatch.setattr(trainer_mod, "dihedral_jacobian", lambda coords, ids: (coords.sum(), ids.sum()))
    monkeypatch.setattr(trainer_mod, "batch_to_atoms", lambda mask, _counts: mask.unsqueeze(-1).expand(-1, -1, 3))

    trainer = FFJointTrainer.__new__(FFJointTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.ffopt_iter = 0
    trainer.logger = MagicMock()

    class FakeModel:
        def _validate_supported_elements(self, _validated_graph):
            raise AssertionError("FFopt should not validate elements")

        def __call__(self, called_graph, skip_ff=False, validate_elements=True):
            model_calls.append(("forward", called_graph, skip_ff, validate_elements))
            return {"energy": torch.tensor([3.0]), "forces": torch.tensor([4.0])}

    trainer.model = FakeModel()

    result = trainer.ffopt(graph, {"pos_res_k": [2.0]})

    assert optimize_calls == [{"pos_res_k": 2.0}]
    assert model_calls == [
        ("forward", graph, False, False),
    ]
    assert torch.allclose(result[0], torch.full((2, 3), 6.0, dtype=torch.float32))
    assert torch.allclose(result[1], graph.coords[1])
