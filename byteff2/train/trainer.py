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

from collections import OrderedDict
import contextlib
from copy import deepcopy
from enum import IntEnum
from glob import glob
import json
import math
import os
import random
import sys
import textwrap
from typing import Any, Union

from git import InvalidGitRepositoryError, NoSuchPathError, Repo
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
import yaml

from byteff2.bytemol.utils import setup_default_logging
from byteff2.data import collate_data, IMDataset, MonoData
from byteff2.model import HybridFF
from byteff2.model.ff_layers.ff_opt import ConstraintFFopt
from byteff2.model.ff_layers.utils import batch_to_atoms, dihedral_jacobian
from byteff2.toolkit.gmxtool import convert_conj_to_bonded_params
from byteff2.train.loss import loss_func, LossType
from byteff2.utils.utilities import get_timestamp


# Loss types whose computation requires autograd (e.g. uses torch.autograd.grad
# or sets requires_grad=True internally). Validation cannot wrap these in
# torch.no_grad()/inference_mode(). Add new autograd-based loss names here.
_AUTOGRAD_REQUIRED_LOSSES = {
    "Partial_Hessian_MAPE",
    "InterEnergyMSE",
    "InterEnergyPolMSE",
    "InterEnergyDispMSE",
    "InterEnergyCTMSE",
    "InterEnergyElecPauliMSE",
}


def _valid_needs_grad(ds_conf: dict) -> bool:
    """Decide whether validation forward + loss for a given dataset must run
    with autograd enabled (vs. inside torch.inference_mode()).

    Returns True when:
      - any configured loss type is in _AUTOGRAD_REQUIRED_LOSSES (e.g.
        Partial_Hessian_MAPE which calls torch.autograd.grad), or
      - the polarizable path is exercised. mm_tspol.MultipoleInt.forward
        uses @torch.enable_grad() + torch.autograd.grad(create_graph=True),
        and torch.enable_grad() CANNOT re-enable grad inside
        torch.inference_mode(). Polarizable runs are gated by `cluster=True`
        in the dataset config.
    """
    loss_names = [d["loss_type"] for d in ds_conf.get("loss", []) or []]
    loss_names += [d["loss_type"] for d in ds_conf.get("aux_loss", []) or []]
    return any(n in _AUTOGRAD_REQUIRED_LOSSES for n in loss_names)


def safe_barrier():
    if dist.is_initialized():
        return dist.barrier()


class TrainState(IntEnum):
    NULL = 0
    STARTED = 1
    FINISHED = 2


class TrainConfig:
    def __init__(self, config: Union[str, dict, Any] = None, timestamp=True, make_working_dir=True, restart=False):

        self._config = {
            "meta": {"work_folder": "", "random_seed": 42, "fp64": False},
            "dataset": [
                {
                    "config": "",
                    "batch_size": 10,
                    "train_ratio": 0.9,
                    "shuffle": True,
                    "loss_weight": 1.0,
                    "loss": {},
                    "aux_loss": {},
                }
            ],
            "model": {},
            "training": {
                "max_epoch": 999,
                "valid_interval": 5,
                "ckpt_interval": 20,
                "optimizer": "Adam",
                "optimizer_params": {},
            },
        }

        custom_config: dict[str, dict] = None

        if isinstance(config, dict):
            custom_config = config
        elif isinstance(config, str):
            with open(config) as file:
                custom_config = yaml.safe_load(file)
        elif config is not None:
            raise TypeError(f"Type {type(config)} is not allowed.")

        if custom_config is not None:
            for k in self._config:
                if k in custom_config:
                    if k == "dataset":
                        defaul_config: dict = self._config[k][0]
                        for i in range(len(custom_config[k])):
                            (new_config := deepcopy(defaul_config)).update(custom_config[k][i])
                            custom_config[k][i] = new_config
                        self._config[k] = custom_config[k]
                    else:
                        self._config[k].update(custom_config[k])

        self.meta: dict = self._config["meta"]
        self.dataset: list[dict] = self._config["dataset"]
        self.model: dict = self._config["model"]
        self.training: dict = self._config["training"]

        self.work_folder = self.meta["work_folder"]
        if timestamp:
            self.work_folder = self.work_folder.rstrip("/") + "_" + get_timestamp()
        self.ckpt_folder = os.path.join(self.work_folder, "ckpt")
        if make_working_dir:
            assert restart or not os.path.exists(self.work_folder), self.work_folder + " already exists."
            os.makedirs(self.ckpt_folder, exist_ok=True)
            self.to_yaml()

    def to_yaml(self, save_path: Union[str, None] = None):
        if save_path is None:
            save_path = os.path.join(self.work_folder, "fftrainer_config_in_use.yaml")
        else:
            assert save_path.endswith(".yaml")
        with open(save_path, "w") as file:
            yaml.dump(self._config, file)

    @property
    def finish_flag(self):
        return os.path.join(self.work_folder, "FINISHED")

    def optimal_path(self, label="") -> str:
        if label:
            return os.path.join(self.work_folder, f"optimal_{label}.pt")
        else:
            return os.path.join(self.work_folder, "optimal.pt")

    def get_latest_ckpt(self) -> str:
        paths = glob(os.path.join(self.ckpt_folder, "ckpt_epoch_*.pt"))
        if not paths:
            return None
        else:
            epochs = [int(fp.split("_")[-1].split(".")[0]) for fp in paths]
            path = os.path.join(self.ckpt_folder, f"ckpt_epoch_{max(epochs)}.pt")
            return path

    @property
    def train_state(self) -> TrainState:
        if not os.path.exists(self.work_folder) or not os.path.exists(self.optimal_path()):
            return TrainState.NULL

        if os.path.exists(self.finish_flag):
            return TrainState.FINISHED

        else:
            return TrainState.STARTED


class FFTrainer:
    def start_ddp(
        self,
        rank,
        world_size,
        device,
        find_unused_parameters=False,
        restart=False,
        load_ckpt=True,
        local_rank=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.local_rank = rank if local_rank is None else local_rank

        self.logger = self._init_logger()
        if self.device.type != "cpu":
            torch.cuda.set_device(self.device)

        dtype = torch.float64 if self.config.meta["fp64"] else torch.float32
        torch.set_default_dtype(dtype)
        if self.rank == 0:
            self.logger.info("set default dtype to %s", dtype)

        self._set_seed(self.config.meta["random_seed"])

        if self.local_rank == 0:
            self.logger.info("loading model")
        ckpt = self.config.model.pop("check_point", None)
        self.model = HybridFF(**self.config.model)
        self.model.to(self.device)

        if world_size > 1:
            # set find_unused_parameters = True when some model parameters a not used in loss
            self.model = DDP(self.model, find_unused_parameters=find_unused_parameters)

        if restart:
            latest_ckpt = self.config.get_latest_ckpt()
            if latest_ckpt:
                self._init_optimizer_scheduler()
                self.load_ckpt(latest_ckpt, model_only=False)
                if self.local_rank == 0:
                    self.logger.info(f"restarted from epoch {self.epoch}")
                self.restarted = True
            safe_barrier()

        if ckpt is not None and not self.restarted and load_ckpt:
            if self.local_rank == 0:
                self.logger.info("loading check point from %s", ckpt)
            self.load_ckpt(ckpt, model_only=True)

        if self.load_data:
            if self.local_rank == 0:
                self.logger.info("loading dataset")
            self.datasets, self.train_dls, self.valid_dls = self._load_data(self.config.dataset)
        else:
            self.datasets, self.train_dls, self.valid_dls = [], [], []

    def __init__(
        self,
        config: Union[str, dict],
        timestamp=True,
        ddp=False,
        device="cuda",
        load_data=True,
        load_shards=None,
        make_working_dir=True,
        restart=False,
        load_ckpt=True,
        use_amp=False,
    ) -> None:
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.config = TrainConfig(config, timestamp, make_working_dir=make_working_dir, restart=restart)
        self.write_log_file = make_working_dir
        self.logger = self._init_logger()
        self.model: Union[HybridFF, DDP] = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None
        self.optimal_state_dict = None
        self.early_stop_count = 0

        self.load_data = load_data
        self.datasets, self.train_dls, self.valid_dls = [], [], []
        self.epoch_step_num = 0

        # training states
        self._init_training_state()
        self._init_extra_training_state()
        self.trainer_state_variables = [
            "epoch",
            "best_valid_loss",
            "early_stop_count",
            "train_history",
            "valid_history",
            "aux_history",
            *self._extra_trainer_state_variables(),
        ]
        self.restarted = False
        self.load_shards = load_shards
        self.use_amp = use_amp
        if self.use_amp and torch.cuda.is_bf16_supported(including_emulation=False):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        if not ddp:
            self.device = torch.device("cuda", 0) if device == "cuda" else torch.device(device)
            self.logger.info(f"using device {self.device}")
            self.start_ddp(self.rank, self.world_size, self.device, restart=restart, load_ckpt=load_ckpt)

    @staticmethod
    def _set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_training_state(self):
        self.epoch = 0
        self.best_valid_loss = torch.finfo().max
        self.early_stop_count = 0
        self.train_history = [[] for _ in self.config.dataset]
        self.valid_history = [[] for _ in self.config.dataset]
        self.aux_history = [[] for _ in self.config.dataset]

    def _init_extra_training_state(self):
        """Hook for subclasses with additional trainer state."""

    def _extra_trainer_state_variables(self) -> list[str]:
        return []

    def _init_optimizer_scheduler(self):
        optim_config = self.config.training["optimizer"].copy()
        optim_type = optim_config.pop("type")

        lr = optim_config.pop("lr")
        if self.local_rank == 0:
            self.logger.info(f"initiating optimizer, lr: {lr}")

        if isinstance(lr, dict):
            parameters = [
                {
                    "params": self.model.module.get_parameters(k)
                    if self.world_size > 1
                    else self.model.get_parameters(k),
                    "lr": v,
                }
                for k, v in lr.items()
            ]
            lr = 1e-3
        else:
            parameters = self.model.parameters()
        self.optimizer = getattr(optim, optim_type)(parameters, lr=lr, **optim_config)

        sched_config = self.config.training.get("scheduler", None)
        if sched_config is not None:
            sched_config = sched_config.copy()
            sched_type = sched_config.pop("type")
            self.scheduler = getattr(optim.lr_scheduler, sched_type)(self.optimizer, **sched_config)

    def _init_logger(self):
        """set logging config for a logger"""
        log_path = os.path.join(self.config.work_folder, "fftrainer.log") if self.write_log_file else None
        logger = setup_default_logging(stdout=True, file_path=log_path)
        if self.local_rank == 0:
            logger.info(f"writing logs to {log_path}")
            try:
                repo = Repo(search_parent_directories=True)
                logger.info(f"current commit: {repo.head.commit.hexsha}")
            except (InvalidGitRepositoryError, NoSuchPathError, ValueError):
                logger.info("current commit: unknown (not running from a git checkout)")
        return logger

    def _train_valid_split(self, dataset: IMDataset, config: dict):
        if data_nums := config.get("data_num", None):
            dataset = dataset[:data_nums]
        # set seed for dataset
        seed = self.config.meta.get("dataset_seed", self.config.meta["random_seed"])
        self._set_seed(seed)
        if shuffle := config.get("shuffle", True):
            dataset = dataset.shuffle()
        train_ratio = config["train_ratio"]
        assert isinstance(train_ratio, float) and 0.0 < train_ratio < 1.0
        train_num = round(len(dataset) * train_ratio)
        if config.get("train_all", False):
            train_ds, valid_ds = dataset, deepcopy(dataset[train_num:])
        else:
            train_ds, valid_ds = dataset[:train_num], dataset[train_num:]

        actual_batch_size = min(config["batch_size"], len(train_ds))
        train_dl = DataLoader(
            train_ds,
            batch_size=actual_batch_size,
            shuffle=shuffle,
            drop_last=len(train_ds) > actual_batch_size,
            collate_fn=collate_data,
            num_workers=config.get("num_workers", 0),
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=collate_data,
        )

        # set seed back
        self._set_seed(self.config.meta["random_seed"])
        return train_dl, valid_dl, max(int(len(train_ds) / actual_batch_size), 1)

    def _load_data(self, dataset_config: list[dict]) -> tuple[list[IMDataset], list[DataLoader], list[DataLoader]]:
        datasets = []
        train_dls, valid_dls, epoch_steps = [], [], []
        for i, config in enumerate(dataset_config):
            if "valid_root" in config:
                # only used in finetuning
                train_dataset = IMDataset(
                    config=config["config"],
                    rank=self.rank,
                    world_size=self.world_size,
                    shard_id=config.get("shard_id", None),
                )
                valid_dataset = IMDataset(
                    config=config["valid_config"], rank=self.rank, world_size=self.world_size, shard_id=None
                )
                train_dl = DataLoader(
                    train_dataset,
                    batch_size=config["batch_size"],
                    shuffle=config.get("shuffle", True),
                    drop_last=True,
                    collate_fn=collate_data,
                    num_workers=config.get("num_workers", 0),
                )
                valid_dl = DataLoader(
                    valid_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_data,
                )
                datasets.append(train_dataset)
                train_dls.append(train_dl)
                valid_dls.append(valid_dl)
                epoch_steps.append(int(len(train_dataset) / config["batch_size"]))
                self.logger.info(f"dataset {i}, num data: {len(train_dataset) + len(valid_dataset)}")
            else:
                dataset = IMDataset(
                    config=config["config"],
                    rank=self.rank,
                    world_size=self.world_size,
                    shard_id=config.get("shard_id", None) if self.load_shards is None else self.load_shards[i],
                )
                datasets.append(dataset)
                train_dl, valid_dl, train_step = self._train_valid_split(dataset, config)
                train_dls.append(train_dl)
                valid_dls.append(valid_dl)
                epoch_steps.append(train_step)
                self.logger.info(
                    f"rank {dataset.rank}, dataset {i}, num data: {len(dataset)}, shard_ids: {dataset.shard_ids}"
                )
        min_step = torch.tensor(max(min(epoch_steps), 1), dtype=torch.int32, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(min_step, op=dist.ReduceOp.MIN)
        self.epoch_step_num = max(min_step.item(), 1)
        return datasets, train_dls, valid_dls

    def save_ckpt(self, save_path: Union[str, None] = None, debug: bool = False):
        if self.rank == 0 or debug:
            if save_path is None:
                ckpt_savepath = os.path.join(self.config.ckpt_folder, f"ckpt_epoch_{self.epoch}.pt")
            else:
                ckpt_savepath = save_path

            self.logger.info(f"saving ckpt to: {ckpt_savepath}")
            sd = {
                "model_state_dict": self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                "optimal_state_dict": self.optimal_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "trainer_state_dict": {name: getattr(self, name) for name in self.trainer_state_variables},
            }
            torch.save(sd, ckpt_savepath)

    def load_ckpt(self, ckpt_path: str, model_only=True):
        sd = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model = self.model.module if self.world_size > 1 else self.model
        try:
            model.load_state_dict(sd["model_state_dict"])
        except RuntimeError as e:
            if self.local_rank == 0:
                self.logger.warning("%s", e)
                self.logger.warning("load state dict failed, use strict=False")
            new_sd = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
            old_sd = sd["model_state_dict"]
            for k in list(old_sd.keys()):
                if k in new_sd and new_sd[k].shape != old_sd[k].shape:
                    old_sd.pop(k)
            model.load_state_dict(old_sd, strict=False)
        if model_only:
            return

        self.optimal_state_dict = sd["optimal_state_dict"]
        self.optimizer.load_state_dict(sd["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(sd["scheduler_state_dict"])

        for k in self.trainer_state_variables:
            setattr(self, k, sd["trainer_state_dict"][k])

    def save_optimal(self):
        optimal_state_dict: dict[str, torch.Tensor] = (
            self.model.module.state_dict().copy() if self.world_size > 1 else self.model.state_dict().copy()
        )
        self.optimal_state_dict = OrderedDict()
        for k, v in optimal_state_dict.items():
            self.optimal_state_dict[k] = v.clone().detach()
        if self.rank == 0:
            torch.save({"model_state_dict": self.optimal_state_dict}, self.config.optimal_path())

    def load_optimal(self):
        if self.optimal_state_dict is not None:
            if self.world_size > 1:
                self.model.module.load_state_dict(self.optimal_state_dict)
            else:
                self.model.load_state_dict(self.optimal_state_dict)

    def calc_loss(self, pred: dict, graph: MonoData, dataset_index: int, is_valid=True) -> list[torch.FloatTensor]:

        loss_types = self.config.dataset[dataset_index]["loss"]
        aux_loss_types = self.config.dataset[dataset_index].get("aux_loss", None)

        losses = [0.0]
        sep_losses = []

        for dct in loss_types:
            name = dct["loss_type"]
            kwargs = dct.get("kwargs", {})
            loss_type = getattr(LossType, name)
            loss = loss_func(pred, graph, loss_type=loss_type, **kwargs)
            sep_losses.append(loss.item())
            weight = dct.get("valid_weight", dct["weight"]) if is_valid else dct["weight"]
            losses[0] += loss * weight

        if aux_loss_types is not None and is_valid:
            for dct in aux_loss_types:
                kwargs = dct.get("kwargs", {})
                name = dct["loss_type"]
                losses.append(loss_func(pred, graph, loss_type=getattr(LossType, name), **kwargs))
        return losses, sep_losses

    def valid_epoch(self):

        safe_barrier()
        self.model.eval()
        self.logger.debug("starting validation epoch")

        averaged_loss = 0.0
        for ids, valid_dl in enumerate(self.valid_dls):
            tot_loss = 0.0
            nbatch = torch.tensor(0, dtype=torch.int64, device=self.device)
            ds_conf = self.config.dataset[ids]

            # Decide whether to disable autograd for this dataset's validation
            # forward + loss. See `_valid_needs_grad` for the full rationale.
            needs_grad = _valid_needs_grad(ds_conf)

            for graph_batch in valid_dl:
                # print(torch.cuda.memory_allocated() / 1024**3)
                # torch.cuda.reset_peak_memory_stats()
                # self.logger.info(f'{graph.name}')
                graph: MonoData = graph_batch.to(self.device)
                nbatch += graph.counts.shape[0]
                ctx = contextlib.nullcontext() if needs_grad else torch.inference_mode()
                with ctx:
                    pred = self.model(
                        graph,
                        skip_ff=ds_conf.get("skip_ff", False),
                        cluster=ds_conf.get("cluster", False),
                        validate_elements=False,
                    )

                    ffparams = pred["ff_parameters"]
                    convert_conj_to_bonded_params(ffparams)
                    losses, _ = self.calc_loss(pred, graph, ids, is_valid=True)

                    # Convert losses to plain python floats inside the
                    # inference_mode context so the resulting tensor is a
                    # regular (non-inference) tensor safe to use afterwards.
                    loss_values = [l.item() for l in losses]
                losses = torch.tensor(loss_values, dtype=torch.float64, device=self.device)
                tot_loss += losses * graph.counts.shape[0]
                del pred, losses

                # print(torch.cuda.max_memory_allocated() / 1024**3)
                # torch.cuda.empty_cache()

            if self.world_size > 1:
                dist.all_reduce(tot_loss)
                dist.all_reduce(nbatch)

            losses = tot_loss / nbatch
            averaged_loss += losses[0] * ds_conf["loss_weight"]

            losses = losses.detach().tolist()
            self.valid_history[ids].append([self.epoch, 0, losses[0]])
            if len(losses) > 1:
                self.aux_history[ids].append([self.epoch, 0] + losses[1:])

            if self.local_rank == 0:
                self.logger.info(f"valid epoch {self.epoch}, dataset {ids}, loss: {losses[0]}")

        if self.local_rank == 0:
            self.logger.info(f"valid epoch {self.epoch} combined loss: {averaged_loss}")

        return averaged_loss

    def train_epoch(self):

        self.model.train()
        self.logger.debug("starting training epoch")
        safe_barrier()
        self.logger.debug(f"train steps: {self.epoch_step_num}, rank: {self.rank}")
        try:
            loaders = [iter(l) for l in self.train_dls]

            for step in range(self.epoch_step_num):
                self.optimizer.zero_grad()

                for ids, loader in enumerate(loaders):
                    ds_conf = self.config.dataset[ids]
                    graph = next(loader).to(self.device)

                    # self.logger.info(f'{graph.name}')
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        pred = self.model(
                            graph,
                            skip_ff=ds_conf.get("skip_ff", False),
                            cluster=ds_conf.get("cluster", False),
                            validate_elements=False,
                        )
                        ffparams = pred["ff_parameters"]
                        convert_conj_to_bonded_params(ffparams)
                        losses, sep_losses = self.calc_loss(pred, graph, ids, is_valid=False)

                    loss = losses[0]
                    if torch.isnan(loss).any():
                        self.logger.warning(f"Found nan in loss! epoch {self.epoch}, step {step}, dataset {ids}, skip!")
                        loss = torch.nan_to_num(loss, nan=0.0)
                    self.train_history[ids].append([self.epoch, step] + [l.item() for l in losses] + sep_losses)
                    if self.local_rank == 0 and step % 10 == 0:
                        self.logger.info(
                            "Train epoch %s, step %s, dataset %s, rank %s, loss: %s",
                            self.epoch,
                            step,
                            ids,
                            self.rank,
                            loss.item(),
                        )
                    loss = loss * ds_conf["loss_weight"]

                    self.amp_scaler.scale(loss).backward()
                    del pred, losses

                safe_barrier()

                if grad_clip := self.config.training.get("grad_clip", None):
                    self.amp_scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), grad_clip)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()

            # average training histories of each process
            if self.world_size > 1:
                safe_barrier()
                new_history = []
                for history_rows in self.train_history:
                    history_tensor = torch.tensor(history_rows, device=self.device)
                    dist.all_reduce(history_tensor, op=dist.ReduceOp.AVG)
                    new_history.append(history_tensor.cpu().tolist())
                self.train_history = new_history

        except KeyboardInterrupt:
            if self.local_rank == 0:
                self.logger.info("stopped by KeyboardInterrupt")
            sys.exit(-1)

    def train_and_valid(self):
        while True:
            if self.epoch % self.config.training["ckpt_interval"] == 0:
                self.save_ckpt()

            # reach max epochs for training iteration
            if self.epoch >= self.config.training["max_epoch"]:
                break

            if self.epoch % self.config.training["valid_interval"] == 0:
                averaged_loss = self.valid_epoch()
                if self.scheduler is not None:
                    self.scheduler.step(averaged_loss)
                    if self.local_rank == 0:
                        for param_group in self.optimizer.param_groups:
                            if "lr" in param_group:
                                self.logger.info(f"current learning rate: {param_group['lr']}")

                if averaged_loss > self.best_valid_loss - self.config.training["ignore_tolerance"]:
                    self.early_stop_count += 1
                else:
                    self.early_stop_count = 0
                    self.best_valid_loss = averaged_loss

                if self.local_rank == 0:
                    self.logger.info(f"early_stop_count: {self.early_stop_count}")

                if averaged_loss <= self.best_valid_loss:
                    self.save_optimal()

                if self.rank == 0:
                    self.plot_history()

                # early stop:
                if self.config.training["early_stop_patience"] <= self.early_stop_count:
                    if self.local_rank == 0:
                        self.logger.info(f"Early stop! Best combined loss: {self.best_valid_loss}")
                    break

            self.train_epoch()
            if self.rank == 0:
                self.plot_history()

            self.epoch += 1

        return self.epoch

    def train_loop(self):
        if self.restarted:
            self.restarted = False
        else:
            self._init_optimizer_scheduler()

        self.train_and_valid()

        with open(self.config.finish_flag, "w"):
            pass

    def plot_history(self):
        history = {"train": self.train_history, "valid": self.valid_history, "aux": self.aux_history}
        with open(os.path.join(self.config.work_folder, "history.json"), "w") as file:
            json.dump(history, file, indent=2)

        plt.cla()
        plt.clf()
        nds = len(self.train_dls)
        fig, axes = plt.subplots(1, nds, figsize=(4 * nds, 3), constrained_layout=True)
        axes = [axes] if nds == 1 else axes.flat

        for ids, ax in enumerate(axes):
            epoch_to_step = OrderedDict()
            epoch_to_step[0] = 0
            for i, res in enumerate(self.train_history[ids]):
                epoch = round(res[0])
                if epoch not in epoch_to_step:
                    epoch_to_step[epoch] = i
            epoch_to_step[max(epoch_to_step) + 1] = len(self.train_history[ids])

            step_to_epoch = OrderedDict()
            for epoch, step in epoch_to_step.items():
                step_to_epoch[step] = epoch

            train_rmse = []
            for i, his in enumerate(self.train_history[ids]):
                epoch = his[0]
                rmse = his[2]
                train_rmse.append([i, rmse])
            train_rmse = np.asarray([[0, np.nan]]) if len(train_rmse) == 0 else np.asarray(train_rmse)
            ax.plot(train_rmse[:, 0], train_rmse[:, 1], label="train")
            if train_rmse[:, 1].max() / train_rmse[:, 1].min() > 20.0:
                ax.semilogy()

            valid_rmse = []
            for epoch, _, rmse in self.valid_history[ids]:
                step = epoch_to_step.get(epoch, len(self.train_history[ids]))
                valid_rmse.append([step, rmse])
            valid_rmse = np.asarray(valid_rmse)
            ax.plot(valid_rmse[:, 0], valid_rmse[:, 1], ".-", label="valid")

            ax.set_xlabel("step")
            ax.grid(visible=True, zorder=1)
            secax = ax.secondary_xaxis("top")
            secax.set_xlabel("epoch")
            epoch_ticks = np.asarray(sorted([(k, v) for k, v in step_to_epoch.items()]))
            if len(epoch_ticks) < 10:
                secax.set_xticks(ticks=epoch_ticks[:, 0], labels=epoch_ticks[:, 1])
            else:
                skip = len(epoch_ticks) // 10 + 1
                secax.set_xticks(ticks=epoch_ticks[::skip, 0], labels=epoch_ticks[::skip, 1])
            up, down = ax.get_ylim()
            ffopt_begin_epoch = getattr(self, "ffopt_begin_epoch", None)
            if ffopt_begin_epoch is not None:
                ax.vlines(
                    [epoch_to_step.get(ep, len(self.train_history[ids])) for ep in ffopt_begin_epoch],
                    down,
                    up,
                    colors="black",
                    linestyle="dashed",
                    zorder=1.5,
                )

            loss_str = " ".join([f"{l['loss_type']}: {l['weight']}" for l in self.config.dataset[ids]["loss"]])
            loss_str = "\n".join(textwrap.wrap(loss_str, width=75))
            ax.set_title(loss_str, fontdict={"fontsize": 8})
            ax.legend(frameon=False, fontsize="small")

            if "aux_loss" in self.config.dataset[ids] and self.config.dataset[ids]["aux_loss"]:
                nax = len(self.config.dataset[ids]["aux_loss"])
                nrows = math.ceil(nax / 3)
                ncol = 3 if nrows > 1 else nax
                aux_fig, aux_axes = plt.subplots(nrows, ncol, figsize=(4 * ncol, 3 * nrows), constrained_layout=True)
                aux_axes = aux_axes.flat if isinstance(aux_axes, np.ndarray) else [aux_axes]
                for ida, aux_ax in enumerate(aux_axes):
                    if ida >= nax:
                        aux_ax.axis("off")
                    else:
                        aux_ax.set_title(self.config.dataset[ids]["aux_loss"][ida]["loss_type"])
                        aux_ax.set_xlabel("epoch")
                        xs, ys = [k[0] for k in self.aux_history[ids]], [k[2 + ida] for k in self.aux_history[ids]]
                        aux_ax.plot(xs, ys, "o-")
                        if min(ys) > 0.0 and max(ys) / min(ys) > 20.0:
                            aux_ax.semilogy()

                aux_fig.suptitle(f"dataset {ids} auxiliary loss")
                aux_fig.savefig(os.path.join(self.config.work_folder, f"aux_history_{ids}.jpg"), dpi=200)
                plt.close(aux_fig)

        fig.savefig(os.path.join(self.config.work_folder, "history.jpg"), dpi=200)
        plt.close(fig)


class FFJointTrainer(FFTrainer):
    def __init__(self, config, timestamp, ddp, restart, use_amp=True):
        super().__init__(config=config, timestamp=timestamp, ddp=ddp, restart=restart, use_amp=use_amp)

    def _init_extra_training_state(self):
        self.ffopt_iter = 0
        self.ffopt_begin_epoch = [0]

    def _extra_trainer_state_variables(self) -> list[str]:
        return ["ffopt_iter", "ffopt_begin_epoch"]

    def ffopt_all(self, ffopt_iter: int):
        for i, config in enumerate(self.config.dataset):
            if "ffopt" in config:
                save_label = f"_ffopt_{ffopt_iter}"
                dataset = self.datasets[i]
                if self.train_dls[i]:
                    self.train_dls[i], self.valid_dls[i] = None, None
                data_num = dataset.load(self.config.work_folder, save_label)
                if data_num > 0:
                    if self.local_rank == 0:
                        self.logger.info(f"loaded ffopt result from {dataset.processed_names[0]}")
                else:
                    if self.local_rank == 0:
                        self.logger.info(f"start ffopt dataset {i}")
                    batch_size = config.get("ffopt_batch_size", 1000)
                    dataset.load()
                    dl = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_data,
                    )
                    for idx, graph in enumerate(dl):
                        coords = self.ffopt(graph, config["ffopt"])
                        dataset.update_data(coords, graph.get_count("node", idx=None), idx * batch_size, "coords")
                        torch.cuda.empty_cache()
                    dataset.save(save_dir=self.config.work_folder, label=save_label)
                    del dl
                    self.logger.info(f"Rank {self.rank}: ffopt dataset {i} done")
                self.train_dls[i], self.valid_dls[i], _ = self._train_valid_split(dataset, config)

    def train_loop(self):
        max_ffopt_iters = self.config.training.get("max_ffopt_iters", 1)
        while self.ffopt_iter < max_ffopt_iters:
            # init optimizer and scheduler, skip if restarted
            if self.restarted:
                self.restarted = False
            else:
                self._init_optimizer_scheduler()

            if self.local_rank == 0:
                self.logger.info(f"ffopt_iter: {self.ffopt_iter}")

            # Set FFopt in eval mode explicitly, then restore the previous mode.
            was_training = self.model.training
            self.model.eval()
            self.ffopt_all(self.ffopt_iter)
            self.model.train(was_training)
            self.train_and_valid()

            self.load_optimal()
            self.ffopt_iter += 1
            self.ffopt_begin_epoch.append(self.epoch)

            self.best_valid_loss = torch.finfo().max
            self.early_stop_count = 0
            self.save_ckpt()

        with open(self.config.finish_flag, "w"):
            pass

    @torch.no_grad()
    def ffopt(self, graph: MonoData, config: dict):
        graph = graph.to(self.device)
        torsion_ids = graph.inc_node_torsion_ids
        model = self.model.module if isinstance(self.model, DDP) else self.model

        def energy_func(_coords):
            graph.coords = _coords
            # Skip element validation throughout FFopt.
            ff_results = model(graph, skip_ff=False, validate_elements=False)
            return ff_results["energy"], ff_results["forces"]

        def jacobian_func(_coords):
            phi, jac = dihedral_jacobian(_coords, torsion_ids)
            return phi, jac

        config = config.copy()
        rk = config["pos_res_k"]
        config["pos_res_k"] = rk[self.ffopt_iter] if isinstance(rk, list) else rk
        opt_coords, converge_flag, niter = ConstraintFFopt.optimize(
            graph,
            energy_func=energy_func,
            jacobian_func=jacobian_func,
            **config,
        )
        self.logger.info(f"Rank {self.rank}: ffopt lbfgs niter: {niter}")

        if opt_coords.isnan().any():
            self.logger.warning(f"converge flag, {torch.where(torch.logical_not(converge_flag))}")
            cc = opt_coords.sum(-1).sum(-1)
            self.logger.warning(f"find nan, {torch.where(cc.isnan())}")
            opt_coords = torch.nan_to_num(opt_coords)

        # if not converged, use init coords for all the confs of the molecule
        if not converge_flag.all():
            ids = set(torch.where(torch.logical_not(converge_flag))[0].tolist())
            self.logger.warning(f"ffopt not converge {len(ids)}/{graph.coords.shape[0]}")
            for idx in ids:
                self.logger.warning(f"ffopt not converge {graph.mol_name[idx]}")
            mask = converge_flag.all(-1, keepdim=True).expand(-1, opt_coords.shape[1])
            mask = batch_to_atoms(mask.to(opt_coords.dtype), graph.get_count("node", idx=None))
            opt_coords = (opt_coords * mask + graph.coords * (1 - mask)).detach()
        return opt_coords
