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

import json
import math
import os
import random
import textwrap
from collections import OrderedDict
from copy import deepcopy
from enum import IntEnum
from glob import glob
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from git import Repo
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from byteff2.data import IMDataset, MonoData, collate_data
from byteff2.model import HybridFF
from byteff2.train.loss import LossType, loss_func
from byteff2.utils.utilities import get_timestamp
from bytemol.utils import setup_default_logging


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
            'meta': {
                'work_folder': '',
                'random_seed': 42,
                'fp64': False
            },
            'dataset': [{
                'config': '',
                'batch_size': 10,
                'train_ratio': 0.9,
                'shuffle': True,
                'loss_weight': 1.0,
                'loss': {},
                'aux_loss': {}
            }],
            'model': {},
            'training': {
                'max_epoch': 999,
                'valid_interval': 5,
                'ckpt_interval': 20,
                'optimizer': 'Adam',
                'optimizer_params': {}
            }
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
                    if k == 'dataset':
                        defaul_config: dict = self._config[k][0]
                        for i in range(len(custom_config[k])):
                            (new_config := deepcopy(defaul_config)).update(custom_config[k][i])
                            custom_config[k][i] = new_config
                        self._config[k] = custom_config[k]
                    else:
                        self._config[k].update(custom_config[k])

        self.meta: dict = self._config['meta']
        self.dataset: list[dict] = self._config['dataset']
        self.model: dict = self._config['model']
        self.training: dict = self._config['training']

        self.work_folder = self.meta['work_folder']
        if timestamp:
            self.work_folder = self.work_folder.rstrip("/") + '_' + get_timestamp()
        self.ckpt_folder = os.path.join(self.work_folder, "ckpt")
        if make_working_dir:
            assert restart or not os.path.exists(self.work_folder), self.work_folder
            os.makedirs(self.ckpt_folder, exist_ok=True)
            self.to_yaml()

    def to_yaml(self, save_path: Union[str, None] = None):
        if save_path is None:
            save_path = os.path.join(self.work_folder, 'fftrainer_config_in_use.yaml')
        else:
            assert save_path.endswith('.yaml')
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file)

    @property
    def finish_flag(self):
        return os.path.join(self.work_folder, 'FINISHED')

    def optimal_path(self, label='') -> str:
        if label:
            return os.path.join(self.work_folder, f'optimal_{label}.pt')
        else:
            return os.path.join(self.work_folder, 'optimal.pt')

    def get_latest_ckpt(self) -> str:
        paths = glob(os.path.join(self.ckpt_folder, "ckpt_epoch_*.pt"))
        if not paths:
            return None
        else:
            epochs = [int(fp.split('_')[-1].split('.')[0]) for fp in paths]
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

    def start_ddp(self, rank, world_size, device, find_unused_parameters=False, restart=False, load_ckpt=True):
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.logger = self._init_logger()
        if device != 'cpu':
            torch.cuda.set_device(device)

        dtype = torch.float64 if self.config.meta['fp64'] else torch.float32
        torch.set_default_dtype(dtype)
        if self.rank == 0:
            self.logger.info('set default dtype to %s', dtype)

        self._set_seed(self.config.meta['random_seed'])

        if self.load_data:
            if self.rank == 0:
                self.logger.info("loading dataset")
            self.datasets, self.train_dls, self.valid_dls = self._load_data(self.config.dataset)
        else:
            self.datasets, self.train_dls, self.valid_dls = [], [], []

        if self.rank == 0:
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
                if self.rank == 0:
                    self.logger.info(f'restarted from epoch {self.epoch}')
                self.restarted = True
            safe_barrier()

        if ckpt is not None and not self.restarted and load_ckpt:
            if self.rank == 0:
                self.logger.info("loading check point from %s", ckpt)
            self.load_ckpt(ckpt, model_only=True)

    def __init__(self,
                 config: Union[str, dict],
                 timestamp=True,
                 ddp=False,
                 device='cuda',
                 load_data=True,
                 load_shards=None,
                 make_working_dir=True,
                 restart=False,
                 load_ckpt=True) -> None:
        self.rank = 0
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
        self.epoch = 0
        self.best_valid_loss = torch.finfo().max
        self.early_stop_count = 0
        self.train_history = [[] for _ in self.config.dataset]
        self.valid_history = [[] for _ in self.config.dataset]
        self.aux_history = [[] for _ in self.config.dataset]
        self.trainer_state_variables = [
            'epoch', 'best_valid_loss', 'early_stop_count', 'train_history', 'valid_history', 'aux_history'
        ]
        self.restarted = False
        self.load_shards = load_shards

        if not ddp:
            self.device = torch.device('cuda', 0) if device == 'cuda' else device
            self.start_ddp(self.rank, self.world_size, self.device, restart=restart, load_ckpt=load_ckpt)

    @staticmethod
    def _set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_optimizer_scheduler(self):
        optim_config = self.config.training['optimizer'].copy()
        optim_type = optim_config.pop('type')

        lr = optim_config.pop('lr')
        if self.rank == 0:
            self.logger.info(f"initiating optimizer, lr: {lr}")

        if isinstance(lr, dict):
            parameters = [{
                'params': self.model.module.get_parameters(k) if self.world_size > 1 else self.model.get_parameters(k),
                'lr': v,
            } for k, v in lr.items()]
            lr = 1e-3
        else:
            parameters = self.model.parameters()
        self.optimizer = getattr(optim, optim_type)(parameters, lr=lr, **optim_config)

        sched_config = self.config.training.get('scheduler', None)
        if sched_config is not None:
            sched_config = sched_config.copy()
            sched_type = sched_config.pop('type')
            self.scheduler = getattr(optim.lr_scheduler, sched_type)(self.optimizer, **sched_config)

    def _init_logger(self):
        """set logging config for a logger"""
        log_path = os.path.join(self.config.work_folder, 'fftrainer.log') if self.write_log_file else None
        logger = setup_default_logging(stdout=True, file_path=log_path)
        if self.rank == 0:
            logger.info(f"writing logs to {log_path}")
            repo = Repo(search_parent_directories=True)
            logger.info(f'current commit: {repo.head.commit.hexsha}')
        return logger

    def _train_valid_split(self, dataset: IMDataset, config: dict):
        if data_nums := config.get('data_num', None):
            dataset = dataset[:data_nums]
        # set seed for dataset
        seed = self.config.meta.get('dataset_seed', self.config.meta['random_seed'])
        self._set_seed(seed)
        if shuffle := config.get('shuffle', True):
            dataset = dataset.shuffle()
        train_ratio = config['train_ratio']
        assert isinstance(train_ratio, float) and 0. < train_ratio < 1.
        train_num = round(len(dataset) * train_ratio)
        if config.get('train_all', False):
            train_ds, valid_ds = dataset, deepcopy(dataset[train_num:])
        else:
            train_ds, valid_ds = dataset[:train_num], dataset[train_num:]

        if (sample_weight := config.get('sample_weight', 1.0)) > 1.0:
            sample_keys = set(config.get('sample_keys', []))
            train_weights = []
            for d in train_ds.data_list:
                name0, name1 = d.name.split('_')[:2]
                if name0 in sample_keys or name1 in sample_keys:
                    train_weights.append(sample_weight)
                else:
                    train_weights.append(1.0)
            valid_weights = []
            for d in valid_ds.data_list:
                name0, name1 = d.name.split('_')[:2]
                if name0 in sample_keys or name1 in sample_keys:
                    # self.logger.info(d.name)
                    valid_weights.append(sample_weight)
                else:
                    valid_weights.append(1.0)

            sampler_train = WeightedRandomSampler(weights=train_weights, replacement=True, num_samples=len(train_ds))
            sampler_valid = WeightedRandomSampler(weights=valid_weights, replacement=True, num_samples=len(valid_ds))
            shuffle = False
            if self.rank == 0:
                self.logger.info(
                    f'Using sampler for training by sample weight: {sample_weight}, sample keys: {sample_keys}')
        else:
            sampler_train = None
            sampler_valid = None
        train_dl = DataLoader(train_ds,
                              batch_size=config['batch_size'],
                              shuffle=shuffle,
                              drop_last=True,
                              collate_fn=collate_data,
                              num_workers=config.get('num_workers', 0),
                              sampler=sampler_train)
        valid_dl = DataLoader(valid_ds,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              drop_last=False,
                              collate_fn=collate_data,
                              sampler=sampler_valid)
        if config.get('record_D_ind', False):
            train_dl.dataset.set_index()
            for data in train_dl.dataset.data_list:
                data['D_ind'] = torch.zeros_like(data['coords'])
                data['D_ind_cluster'] = torch.zeros_like(data['coords'])
            valid_dl.dataset.set_index()
            for data in valid_dl.dataset.data_list:
                data['D_ind'] = torch.zeros_like(data['coords'])
                data['D_ind_cluster'] = torch.zeros_like(data['coords'])

        # set seed back
        self._set_seed(self.config.meta['random_seed'])
        return train_dl, valid_dl, int(len(train_ds) / config['batch_size'])

    def _load_data(self, dataset_config: list[dict]) -> tuple[list[IMDataset], list[DataLoader], list[DataLoader]]:
        datasets = []
        train_dls, valid_dls, epoch_steps = [], [], []
        for i, config in enumerate(dataset_config):

            if 'valid_root' in config:
                # only used in finetuning
                train_dataset = IMDataset(config=config['config'],
                                          rank=self.rank,
                                          world_size=self.world_size,
                                          shard_id=config.get('shard_id', None))
                valid_dataset = IMDataset(config=config['valid_config'],
                                          rank=self.rank,
                                          world_size=self.world_size,
                                          shard_id=None)
                train_dl = DataLoader(train_dataset,
                                      batch_size=config['batch_size'],
                                      shuffle=config.get('shuffle', True),
                                      drop_last=True,
                                      collate_fn=collate_data,
                                      num_workers=config.get('num_workers', 0))
                valid_dl = DataLoader(valid_dataset,
                                      batch_size=config['batch_size'],
                                      shuffle=False,
                                      drop_last=False,
                                      collate_fn=collate_data)
                datasets.append(train_dataset)
                train_dls.append(train_dl)
                valid_dls.append(valid_dl)
                epoch_steps.append(int(len(train_dataset) / config['batch_size']))
                self.logger.info(f"dataset {i}, num data: {len(train_dataset)+len(valid_dataset)}")
            else:
                dataset = IMDataset(
                    config=config['config'],
                    rank=self.rank,
                    world_size=self.world_size,
                    shard_id=config.get('shard_id', None) if self.load_shards is None else self.load_shards[i])
                datasets.append(dataset)
                train_dl, valid_dl, train_step = self._train_valid_split(dataset, config)
                train_dls.append(train_dl)
                valid_dls.append(valid_dl)
                epoch_steps.append(train_step)
                self.logger.info(
                    f"rank {dataset.rank}, dataset {i}, num data: {len(dataset)}, shard_ids: {dataset.shard_ids}")
        min_step = torch.tensor(min(epoch_steps), dtype=torch.int32, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(min_step, op=dist.ReduceOp.MIN)
        self.epoch_step_num = min_step.item()
        return datasets, train_dls, valid_dls

    def save_ckpt(self, save_path: Union[str, None] = None, debug: bool = False):
        if self.rank == 0 or debug:
            if save_path is None:
                ckpt_savepath = os.path.join(self.config.ckpt_folder, f"ckpt_epoch_{self.epoch}.pt")
            else:
                ckpt_savepath = save_path

            self.logger.info(f'saving ckpt to: {ckpt_savepath}')
            sd = {
                'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                'optimal_state_dict': self.optimal_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'trainer_state_dict': {
                    name: getattr(self, name) for name in self.trainer_state_variables
                },
            }
            torch.save(sd, ckpt_savepath)

    def load_ckpt(self, ckpt_path: str, model_only=True):
        sd = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model = self.model.module if self.world_size > 1 else self.model
        try:
            model.load_state_dict(sd['model_state_dict'])
        except RuntimeError as e:
            if self.rank == 0:
                self.logger.warning("%s", e)
                self.logger.warning("load state dict failed, use strict=False")
            new_sd = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
            old_sd = sd['model_state_dict']
            for k in list(old_sd.keys()):
                if k in new_sd and new_sd[k].shape != old_sd[k].shape:
                    old_sd.pop(k)
            model.load_state_dict(old_sd, strict=False)
        if model_only:
            return

        self.optimal_state_dict = sd['optimal_state_dict']
        self.optimizer.load_state_dict(sd['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(sd['scheduler_state_dict'])

        for k in self.trainer_state_variables:
            setattr(self, k, sd['trainer_state_dict'][k])

    def save_optimal(self):
        optimal_state_dict: dict[str, torch.Tensor] = self.model.module.state_dict().copy(
        ) if self.world_size > 1 else self.model.state_dict().copy()
        self.optimal_state_dict = OrderedDict()
        for k, v in optimal_state_dict.items():
            self.optimal_state_dict[k] = v.clone().detach()
        if self.rank == 0:
            torch.save({'model_state_dict': self.optimal_state_dict}, self.config.optimal_path())

    def load_optimal(self):
        if self.optimal_state_dict is not None:
            if self.world_size > 1:
                self.model.module.load_state_dict(self.optimal_state_dict)
            else:
                self.model.load_state_dict(self.optimal_state_dict)

    def calc_loss(self, pred: dict, graph: MonoData, dataset_index: int, is_valid=True) -> list[torch.FloatTensor]:

        loss_types = self.config.dataset[dataset_index]['loss']
        aux_loss_types = self.config.dataset[dataset_index].get('aux_loss', None)

        losses = [0.]

        for dct in loss_types:
            name = dct['loss_type']
            kwargs = dct.get('kwargs', {})
            loss_type = getattr(LossType, name)
            # print(loss_type)
            loss = loss_func(pred, graph, loss_type=loss_type, **kwargs)
            weight = dct.get('valid_weight', dct['weight']) if is_valid else dct['weight']
            losses[0] += loss * weight

        if aux_loss_types is not None and is_valid:
            for dct in aux_loss_types:
                kwargs = dct.get('kwargs', {})
                name = dct['loss_type']
                losses.append(loss_func(pred, graph, loss_type=getattr(LossType, name), **kwargs))
                # print(name, kwargs, losses[-1])
        return losses

    def valid_epoch(self):

        safe_barrier()
        self.model.eval()
        self.logger.debug('starting validation epoch')

        averaged_loss = 0.
        for ids, valid_dl in enumerate(self.valid_dls):

            tot_loss = 0.
            nbatch = torch.tensor(0, dtype=torch.int64, device=self.device)
            ds_conf = self.config.dataset[ids]
            for graph in valid_dl:
                # print(torch.cuda.memory_allocated() / 1024**3)
                # torch.cuda.reset_peak_memory_stats()
                # self.logger.info(f'{graph.name}')
                graph: MonoData = graph.to(self.device)
                nbatch += graph.counts.shape[0]
                pred = self.model(graph, skip_ff=ds_conf.get('skip_ff', False), cluster=ds_conf.get('cluster', False))

                # record D_ind for training efficiency
                if ds_conf.get('record_D_ind', False):
                    D_ind = pred['ff_parameters']['MultipoleInt.D_ind'].clone().detach().to('cpu')
                    D_ind_cluster = pred['ff_parameters']['MultipoleInt.D_ind_cluster'].clone().detach().to('cpu')
                    shift = 0
                    for idx in graph['data_idx']:
                        data = valid_dl.dataset.data_list[idx]
                        assert idx == data['data_idx']
                        na = data.coords.shape[0]
                        data['D_ind'] = D_ind[shift:shift + na]
                        data['D_ind_cluster'] = D_ind_cluster[shift:shift + na]
                        shift += na

                with torch.no_grad():
                    losses = self.calc_loss(pred, graph, ids, is_valid=True)
                    losses = torch.tensor([l.item() for l in losses], dtype=torch.float64, device=self.device)
                # if losses[1] > 1e6:
                #     self.logger.info(f"losses: {losses}")
                tot_loss += losses * graph.counts.shape[0]
                del pred, losses

                # print(torch.cuda.max_memory_allocated() / 1024**3)
                # torch.cuda.empty_cache()

            if self.world_size > 1:
                dist.all_reduce(tot_loss)
                dist.all_reduce(nbatch)

            losses = tot_loss / nbatch
            averaged_loss += losses[0] * ds_conf['loss_weight']

            losses = losses.detach().tolist()
            self.valid_history[ids].append([self.epoch, 0, losses[0]])
            if len(losses) > 1:
                self.aux_history[ids].append([self.epoch, 0] + losses[1:])

            if self.rank == 0:
                self.logger.info(f'valid epoch {self.epoch}, dataset {ids}, loss: {losses[0]}')

        if self.rank == 0:
            self.logger.info(f'valid epoch {self.epoch} combined loss: {averaged_loss}')

        return averaged_loss

    def train_epoch(self):

        self.model.train()
        self.logger.debug('starting training epoch')
        safe_barrier()
        self.logger.debug(f'train steps: {self.epoch_step_num}, rank: {self.rank}')
        try:
            loaders = [iter(l) for l in self.train_dls]

            for step in range(self.epoch_step_num):

                self.optimizer.zero_grad()

                for ids, loader in enumerate(loaders):
                    ds_conf = self.config.dataset[ids]
                    graph = next(loader).to(self.device)
                    # self.logger.info(f'{graph.name}')
                    pred = self.model(graph,
                                      skip_ff=ds_conf.get('skip_ff', False),
                                      cluster=ds_conf.get('cluster', False))
                    # record D_ind for training efficiency
                    if ds_conf.get('record_D_ind', False):
                        D_ind = pred['ff_parameters']['MultipoleInt.D_ind'].clone().detach().to('cpu')
                        D_ind_cluster = pred['ff_parameters']['MultipoleInt.D_ind_cluster'].clone().detach().to('cpu')
                        # print(graph.coords.shape, D_ind.shape)
                        shift = 0
                        for idx in graph['data_idx']:
                            data = self.train_dls[ids].dataset.data_list[idx]
                            assert idx == data['data_idx']
                            na = data.coords.shape[0]
                            data['D_ind'] = D_ind[shift:shift + na]
                            data['D_ind_cluster'] = D_ind_cluster[shift:shift + na]
                            assert data[
                                'D_ind'].shape == data.coords.shape, f"{data['D_ind'].shape} != {data.coords.shape}"
                            assert data[
                                'D_ind_cluster'].shape == data.coords.shape, f"{data['D_ind_cluster'].shape} != {data.coords.shape}"
                            shift += na

                    losses = self.calc_loss(pred, graph, ids, is_valid=False)
                    loss = losses[0]
                    if torch.isnan(loss).any():
                        raise RuntimeError('Found nan in loss!')
                    self.train_history[ids].append([self.epoch, step, loss.item()])
                    self.logger.info(
                        f'Train epoch {self.epoch}, step {step}, dataset {ids}, rank {self.rank}, loss: {loss.item()}')
                    loss = loss * ds_conf['loss_weight']

                    if self.config.meta.get('save_debug_data', False) and self.rank == 0:
                        if len(self.train_history[ids]) > 1 and loss.item() > 1.5 * self.train_history[ids][-2][-1]:
                            save_dir = os.path.join(self.config.work_folder, "debug")
                            os.makedirs(save_dir, exist_ok=True)
                            self.save_ckpt(os.path.join(save_dir, f"ckpt_{self.epoch}_{step}.pt"), debug=True)
                            torch.save(graph, os.path.join(save_dir, f"data_{self.epoch}_{step}.pt"))
                            self.logger.info("debug data saved!")
                    loss.backward()
                    del pred, losses

                safe_barrier()

                if grad_clip := self.config.training.get('grad_clip', None):
                    clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            # average training histories of each process
            if self.world_size > 1:
                train_history = torch.tensor(self.train_history, device=self.device)
                dist.all_reduce(train_history, op=dist.ReduceOp.AVG)
                self.train_history = train_history.cpu().tolist()

        except KeyboardInterrupt:
            if self.rank == 0:
                self.logger.info('stopped by KeyboardInterrupt')
            exit(-1)

    def train_loop(self):
        # self.model.preff_block.pre_layers['PreExp6CT'].ct_eps_mlp.reset_parameters()
        # self.model.preff_block.pre_layers['PreExp6CT'].c6_mlp.reset_parameters()

        # init optimizer and scheduler, skip if restarted
        if self.restarted:
            self.restarted = False
        else:
            self._init_optimizer_scheduler()

        while True:

            if self.epoch % self.config.training['ckpt_interval'] == 0:
                self.save_ckpt()

            # reach max epochs for training iteration
            if self.epoch >= self.config.training['max_epoch']:
                break

            if self.epoch % self.config.training['valid_interval'] == 0:
                averaged_loss = self.valid_epoch()
                if self.scheduler is not None:
                    self.scheduler.step(averaged_loss)
                    if self.rank == 0:
                        self.logger.info(f"current learning rate: {self.optimizer.param_groups[0]['lr']}")

                if averaged_loss > self.best_valid_loss - self.config.training['ignore_tolerance']:
                    self.early_stop_count += 1
                else:
                    self.early_stop_count = 0
                    self.best_valid_loss = averaged_loss

                if self.rank == 0:
                    self.logger.info(f'early_stop_count: {self.early_stop_count}')

                if averaged_loss <= self.best_valid_loss:
                    self.save_optimal()

                if self.rank == 0:
                    self.plot_history()

                # early stop:
                if self.config.training['early_stop_patience'] <= self.early_stop_count:
                    if self.rank == 0:
                        self.logger.info(f"Early stop! Best combined loss: {self.best_valid_loss}")
                    break

            self.train_epoch()
            if self.rank == 0:
                self.plot_history()

            self.epoch += 1

        with open(self.config.finish_flag, "w"):
            pass

        return self.epoch

    def plot_history(self):
        history = {"train": self.train_history, "valid": self.valid_history, "aux": self.aux_history}
        with open(os.path.join(self.config.work_folder, 'history.json'), "w") as file:
            json.dump(history, file)

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
            for i, (epoch, _, rmse) in enumerate(self.train_history[ids]):
                train_rmse.append([i, rmse])
            train_rmse = np.asarray([[0, np.nan]]) if len(train_rmse) == 0 else np.asarray(train_rmse)
            ax.plot(train_rmse[:, 0], train_rmse[:, 1], label='train')
            if train_rmse[:, 1].max() / train_rmse[:, 1].min() > 20.:
                ax.semilogy()

            valid_rmse = []
            for epoch, _, rmse in self.valid_history[ids]:
                valid_rmse.append([epoch_to_step[epoch], rmse])
            valid_rmse = np.asarray(valid_rmse)
            ax.plot(valid_rmse[:, 0], valid_rmse[:, 1], '.-', label='valid')

            ax.set_xlabel('step')
            ax.grid(visible=True, zorder=1)
            secax = ax.secondary_xaxis('top')
            secax.set_xlabel('epoch')
            epoch_ticks = np.asarray(sorted([(k, v) for k, v in step_to_epoch.items()]))
            if len(epoch_ticks) < 10:
                secax.set_xticks(ticks=epoch_ticks[:, 0], labels=epoch_ticks[:, 1])
            else:
                skip = len(epoch_ticks) // 10 + 1
                secax.set_xticks(ticks=epoch_ticks[::skip, 0], labels=epoch_ticks[::skip, 1])
            loss_str = ' '.join([f"{l['loss_type']}: {l['weight']}" for l in self.config.dataset[ids]['loss']])
            loss_str = '\n'.join(textwrap.wrap(loss_str, width=75))
            ax.set_title(loss_str, fontdict={"fontsize": 8})
            ax.legend(frameon=False, fontsize="small")

            if 'aux_loss' in self.config.dataset[ids] and self.config.dataset[ids]['aux_loss']:
                nax = len(self.config.dataset[ids]['aux_loss'])
                nrows = math.ceil(nax / 3)
                ncol = 3 if nrows > 1 else nax
                aux_fig, aux_axes = plt.subplots(nrows, ncol, figsize=(4 * ncol, 3 * nrows), constrained_layout=True)
                aux_axes = aux_axes.flat if isinstance(aux_axes, np.ndarray) else [aux_axes]
                for ida, aux_ax in enumerate(aux_axes):
                    if ida >= nax:
                        aux_ax.axis('off')
                    else:
                        aux_ax.set_title(self.config.dataset[ids]['aux_loss'][ida]['loss_type'])
                        aux_ax.set_xlabel('epoch')
                        xs, ys = [k[0] for k in self.aux_history[ids]], [k[2 + ida] for k in self.aux_history[ids]]
                        aux_ax.plot(xs, ys, 'o-')
                        if min(ys) > 0. and max(ys) / min(ys) > 20.:
                            aux_ax.semilogy()

                aux_fig.suptitle(f"dataset {ids} auxiliary loss")
                aux_fig.savefig(os.path.join(self.config.work_folder, f'aux_history_{ids}.jpg'), dpi=200)
                plt.close(aux_fig)

        fig.savefig(os.path.join(self.config.work_folder, 'history.jpg'), dpi=200)
        plt.close(fig)

    def compute_ewc_fisher(self, idataset=0):
        self.model.eval()

        fisher = {}
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            fisher[name] = torch.zeros_like(param.data)

        ds_conf = self.config.dataset[idataset]
        nsample = len(self.datasets[idataset])
        for i, data in enumerate(self.datasets[idataset]):
            self.model.zero_grad()
            graph = collate_data([data]).to(self.device)
            pred = self.model(graph, skip_ff=ds_conf.get('skip_ff', False), cluster=ds_conf.get('cluster', False))
            losses = self.calc_loss(pred, graph, idataset, is_valid=False)
            loss = losses[0]
            if i % 100 == 0:
                self.logger.info(f'step {i}, loss: {loss.item():.2f}')
            if torch.isnan(loss).any():
                raise RuntimeError('Found nan in loss!')
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                fisher[name] += param.grad**2 / nsample
        return fisher
