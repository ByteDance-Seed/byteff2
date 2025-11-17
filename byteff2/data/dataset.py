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

import copy
import json
import logging
import os
import random
import traceback
from typing import Any, TypeVar, Union

import h5py
import torch
import yaml
from torch.utils.data import Dataset

from byteff2.data import Data
from byteff2.data import data as bdata
from byteff2.utils.utilities import get_timestamp

logger = logging.getLogger(__name__)


class DatasetConfig:

    def __init__(self, config: Union[str, dict, Any] = None):

        # default config
        self._config = {
            'meta_fp': '',  # containing information of raw h5 and json
            'save_dir': '',  # path to save converted pkl
            'data_cls': 'Data',  # any subclass of byteff2.data.Data
            'shards': 1,  # number of shards to save
            'confdata': {},  # argument: dataset name
            'moldata': {},
            'kwargs': {}  # other kwargs for Data
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
                    self._config[k] = copy.deepcopy(custom_config[k])

    def __str__(self) -> str:
        return self._config.__str__()

    def get(self, name: str):
        return self._config[name]

    def to_yaml(self, save_path=None, timestamp=True):
        if save_path is None:
            save_path = os.path.join(self.get('save_dir'), 'dataset_config.yaml')
        else:
            assert save_path.endswith('.yaml')

        if timestamp:
            self._config['timestamp'] = get_timestamp()

        with open(save_path, 'w') as file:
            yaml.dump(self._config, file)


T = TypeVar('T', bound=Data)


class IMDataset(Dataset[T]):

    def __init__(self,
                 config: Union[str, dict],
                 rank: int = 0,
                 world_size: int = 1,
                 shard_id: Union[int, Any] = None,
                 processing=False):
        super().__init__()

        self.config = DatasetConfig(config)
        if isinstance(config, str):
            self.save_dir = os.path.dirname(config)
        else:
            self.save_dir = self.config.get('save_dir')
        self.rank = rank
        self.world_size = world_size
        self.shards = self.config.get('shards')

        if shard_id is None:
            shard_ids = list(range(self.shards))
        elif isinstance(shard_id, int):
            assert 0 <= shard_id < self.shards
            shard_ids = [shard_id]
        else:
            assert all([0 <= s < self.shards and isinstance(s, int) for s in shard_id])
            shard_ids = list(shard_id)

        # padding shard_ids to integer multiples of world_size
        if len(shard_ids) % world_size:
            target_len = (len(shard_ids) // world_size + 1) * world_size
            shard_ids = (shard_ids * (target_len // len(shard_ids) + 1))[:target_len]

        self.shard_ids = shard_ids[rank::world_size]

        self.data_list: list[T] = []

        if not processing:
            self.check_exist()
            self.load()

    def copy(self) -> 'IMDataset':
        new_dataset = IMDataset(config=self.config._config,
                                rank=self.rank,
                                world_size=self.world_size,
                                processing='skip')
        new_dataset.shard_ids = self.shard_ids.copy()
        new_dataset.data_list = self.data_list.copy()
        return new_dataset

    def __len__(self):
        return len(self.data_list)

    @property
    def processed_names(self) -> list[str]:
        return [os.path.join(self.save_dir, f'processed_data_shard{shard_id}.pkl') for shard_id in self.shard_ids]

    def check_exist(self):
        assert all([os.path.exists(name) for name in self.processed_names
                   ]), [os.path.exists(name) for name in self.processed_names]

    def __getitem__(self, index: Union[int, slice]) -> Union[T, list[T]]:
        if isinstance(index, int):
            return self.data_list[index]
        elif isinstance(index, slice):
            ret = self.copy()
            ret.data_list = ret.data_list[index]
            ret.set_index()
            return ret
        else:
            raise TypeError(f'index of type {type(index)} is not allowed.')

    def shuffle(self):
        ret = self.copy()
        random.shuffle(ret.data_list)
        ret.set_index()
        return ret

    def set_index(self):
        for i, data in enumerate(self.data_list):
            data['data_idx'] = i

    def load(self):
        self.data_list = []
        data_cls: Data = getattr(bdata, self.config.get('data_cls'))
        for fname in self.processed_names:
            self.data_list += [data_cls.from_dict(d) for d in torch.load(fname, weights_only=True)]
        self.set_index()

    @classmethod
    def process(cls, config: str, shard_id: int):

        ds = cls(config, processing=True)
        if 'local_frames' in ds.config._config['kwargs']:
            with open(ds.config._config['kwargs']['local_frames']) as file:
                local_frames = json.load(file)
            ds.config._config['kwargs']['local_frames'] = local_frames

        logger.info(f'processing shard {shard_id}')

        meta = ds.config.get('meta_fp')
        with open(meta) as file:
            lines = file.readlines()
            lines = lines[shard_id::ds.shards]

        data_cls = getattr(bdata, ds.config.get('data_cls'))
        assert issubclass(data_cls, Data)

        root = os.path.dirname(meta)
        h5_dict = {}
        name_mps_dict = {}
        data_list = []
        for line in lines:
            if len(data_list) % 1000 == 0:
                logger.info(f'finished mol {len(data_list)}')

            args = line.rstrip().split(',')
            if len(args) != 2:
                continue
            dataset_name, name = args

            if dataset_name not in h5_dict:
                h5_dict[dataset_name] = h5py.File(os.path.join(root, dataset_name + '.h5'), 'r')
                with open(os.path.join(root, dataset_name + '.json'), 'r') as file:
                    name_mps_dict[dataset_name] = json.load(file)

            confdata = {k: h5_dict[dataset_name][name][v][:] for k, v in ds.config.get('confdata').items()}
            moldata = {k: h5_dict[dataset_name][name][v][:] for k, v in ds.config.get('moldata').items()}
            try:
                data = data_cls(name=name,
                                mapped_smiles=name_mps_dict[dataset_name][name],
                                confdata=confdata,
                                moldata=moldata,
                                **ds.config.get('kwargs'))
            except:  # pylint: disable=bare-except
                logger.warning(f'failed: {name}, skip!')
                logger.warning(traceback.format_exc())
                continue

            data_list.append(data)

        os.makedirs(os.path.dirname(ds.processed_names[shard_id]), exist_ok=True)
        torch.save([dict(d) for d in data_list], ds.processed_names[shard_id])
        ds.config.to_yaml()

        for file in h5_dict.values():
            file.close()

        logger.info(f'finished shard {shard_id}')
