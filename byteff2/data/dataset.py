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

from abc import ABC, abstractmethod
import copy
import csv
import json
import logging
import os
import random
import traceback
from typing import Any, Iterable, TypeVar, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from byteff2.data import Data, data as bdata
from byteff2.utils.utilities import get_timestamp


# from byteff2.bytemol.utils.tar import MolTarLoader

logger = logging.getLogger(__name__)

_MISSING_CSV_VALUES = {"", "#n/a", "#na", "<na>", "n/a", "na", "nan", "nat", "none", "null"}


def _read_h5_dataset(dataset):
    """Read h5 dataset, handling scalar data correctly."""
    if dataset.shape == ():
        return np.array(dataset[()])
    return dataset[:]


def _is_missing_csv_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_CSV_VALUES
    return False


class DatasetConfig:
    def __init__(self, config: Union[str, dict, Any] = None):

        # default config
        self._config = {
            "meta_fp": "",  # containing information of raw h5 and json
            "meta_asset": "",  # release-relative metadata path resolved by example entry points
            "save_dir": "",  # path to save converted pkl
            "data_cls": "Data",  # any subclass of byteff2.data.Data
            "shards": 1,  # number of shards to save
            "confdata": {},  # argument: dataset name
            "moldata": {},
            "kwargs": {},  # other kwargs for Data
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

    def set(self, name: str, value: Any):
        self._config[name] = value

    def to_yaml(self, save_path=None, timestamp=True):
        if save_path is None:
            save_path = os.path.join(self.get("save_dir"), "dataset_config.yaml")
        else:
            assert save_path.endswith(".yaml")

        if timestamp:
            self._config["timestamp"] = get_timestamp()

        with open(save_path, "w") as file:
            yaml.dump(self._config, file)


T = TypeVar("T", bound=Data)


class DatasetProcessor(ABC):
    """Abstract dataset preprocessing contract.

    A processor owns the dataset-specific parts of raw-data traversal and sample
    construction, while ``IMDataset.process`` owns the generic shard loop,
    per-sample persistence flow, and the default exception isolation around
    ``process_input``.

    Responsibility boundary:
    - ``iter_inputs`` / ``get_shard_inputs`` enumerate candidate raw inputs for a
      shard. If these methods perform extra I/O or decoding, the processor should
      catch item-level errors itself and skip bad samples there, otherwise a
      generator exception will terminate the remaining shard iteration.
    - ``process_input`` converts one prepared input into a ``Data`` instance.
      Exceptions raised here are treated as item-level failures by
      ``IMDataset.process`` and will be logged then skipped.
    - Returning ``None`` from ``process_input`` means the sample is intentionally
      skipped without raising an additional error.
    - ``finalize`` is responsible for releasing processor-owned resources such as
      file handles, caches, or temporary state, and will be called once after the
      shard loop exits.
    """

    def get_shard_inputs(self, ds: "IMDataset", shard_id: int) -> Iterable[Any]:
        for idx, item in enumerate(self.iter_inputs(ds)):
            if idx % ds.shards == shard_id:
                yield item

    @abstractmethod
    def iter_inputs(self, ds: "IMDataset") -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def process_input(self, ds: "IMDataset", item: Any) -> Data | None:
        raise NotImplementedError

    def describe_item(self, item: Any) -> str:
        if isinstance(item, dict):
            for key in ["uuid", "name", "id"]:
                value = item.get(key)
                if value is not None:
                    return str(value)
        return str(item)

    def finalize(self, ds: "IMDataset"):
        return None


class ByteFFDatasetProcessor(DatasetProcessor):
    def __init__(self):
        self._root = None
        self._is_legacy_meta = None
        self._meta_fp = None
        self._legacy_smiles_cache: dict[str, dict] = {}
        self._h5_dict: dict[str, h5py.File] = {}
        self._data_cls = None

    def _prepare(self, ds: "IMDataset"):
        local_frames = ds.config._config["kwargs"].get("local_frames")
        if isinstance(local_frames, str):
            with open(local_frames) as file:
                ds.config._config["kwargs"]["local_frames"] = json.load(file)

        if self._meta_fp is not None:
            return

        self._meta_fp = ds.config.get("meta_fp")
        self._root = os.path.dirname(self._meta_fp)

        with open(self._meta_fp) as meta_file:
            first_line = meta_file.readline()
        first_cols = [c.strip() for c in first_line.split(",")]
        self._is_legacy_meta = "uuid" not in first_cols and "h5_file" not in first_cols

        self._data_cls = getattr(bdata, ds.config.get("data_cls"))
        assert issubclass(self._data_cls, Data)

    def iter_inputs(self, ds: "IMDataset"):
        self._prepare(ds)
        if self._is_legacy_meta:
            with open(self._meta_fp, newline="") as meta_file:
                reader = csv.reader(meta_file)
                for row in reader:
                    if not row:
                        continue
                    yield {"dataset_name": row[0], "name": row[1]}
            return

        with open(self._meta_fp, newline="") as meta_file:
            reader = csv.DictReader(meta_file)
            for row in reader:
                yield row

    def get_shard_inputs(self, ds: "IMDataset", shard_id: int):
        for item in super().get_shard_inputs(ds, shard_id):
            uuid = item.get("uuid", item.get("name", "unknown"))
            try:
                if self._is_legacy_meta:
                    dataset_name = item["dataset_name"]
                    uuid = item["name"]
                    h5_file = f"{dataset_name}.h5"
                    mapped_smiles = self._legacy_get_smiles(dataset_name, uuid)
                else:
                    uuid = item["uuid"]
                    h5_file = item["h5_file"]
                    mapped_smiles = item.get("mapped_isomeric_smiles")

                if _is_missing_csv_value(mapped_smiles):
                    raise ValueError(f"missing mapped_isomeric_smiles for uuid={uuid} (h5_file={h5_file})")

                if h5_file not in self._h5_dict:
                    self._h5_dict[h5_file] = h5py.File(os.path.join(self._root, h5_file), "r")

                yield {
                    "uuid": uuid,
                    "mapped_smiles": mapped_smiles,
                    "h5_group": self._h5_dict[h5_file][uuid],
                }
            except Exception:  # pylint: disable=broad-except
                logger.warning(f"failed: {uuid}, skip!")
                logger.warning(traceback.format_exc())
                continue

    def _legacy_get_smiles(self, dataset_name: str, name: str):
        if dataset_name not in self._legacy_smiles_cache:
            json_fp = os.path.join(self._root, f"{dataset_name}.json")
            if os.path.exists(json_fp):
                with open(json_fp) as json_file:
                    self._legacy_smiles_cache[dataset_name] = json.load(json_file)
            else:
                self._legacy_smiles_cache[dataset_name] = {}
        return self._legacy_smiles_cache[dataset_name].get(name)

    def process_input(self, ds: "IMDataset", item: dict):
        uuid = item["uuid"]
        mapped_smiles = item["mapped_smiles"]
        h5_group = item["h5_group"]
        constraint_keys = sorted(
            [k for k in h5_group.keys() if k.startswith("constraint")], key=lambda x: int(x.split()[1])
        )

        if constraint_keys:
            confdata = {}
            for field, h5_key in ds.config.get("confdata").items():
                values = []
                for constraint_key in constraint_keys:
                    value = _read_h5_dataset(h5_group[constraint_key][h5_key])
                    if value.ndim == 0:
                        value = value.reshape(1)
                    values.append(value)
                if values[0].ndim >= 2:
                    confdata[field] = np.stack(values, axis=0)
                else:
                    confdata[field] = np.concatenate(values, axis=0)

            moldata = {}
            for key, h5_key in ds.config.get("moldata").items():
                value = _read_h5_dataset(h5_group[h5_key])
                if value.ndim == 1 and key == "torsion_ids":
                    value = value.reshape(1, -1)
                moldata[key] = value
        else:
            confdata = {}
            for key, h5_key in ds.config.get("confdata").items():
                value = _read_h5_dataset(h5_group[h5_key])
                if value.ndim == 2:
                    value = value[np.newaxis]
                confdata[key] = value
            moldata = {key: _read_h5_dataset(h5_group[h5_key]) for key, h5_key in ds.config.get("moldata").items()}

        confdata, moldata = self.postprocess_raw_data(confdata, moldata)

        return self._data_cls(
            name=uuid,
            mapped_smiles=mapped_smiles,
            confdata=confdata,
            moldata=moldata,
            **ds.config.get("kwargs"),
        )

    def postprocess_raw_data(self, confdata: dict, moldata: dict):
        """Hook to adjust raw numpy conf/mol data before Data construction.

        Runs while ``confdata``/``moldata`` are still plain numpy arrays at their
        h5-stored precision (typically float64) and before the float32 cast in
        ``Data.__init__``. Subclasses can override this to, e.g., mean-center
        per-molecule energies in float64. The default is a no-op.
        """
        return confdata, moldata

    def finalize(self, ds: "IMDataset"):
        del ds
        for file in self._h5_dict.values():
            file.close()
        self._h5_dict.clear()


class IMDataset(Dataset[T]):
    def __init__(
        self,
        config: Union[str, dict],
        rank: int = 0,
        world_size: int = 1,
        shard_id: Union[int, Any] = None,
        processing=False,
    ):
        super().__init__()

        self.config = DatasetConfig(config)
        if isinstance(config, str):
            # Shards are co-located with the config yaml (see DatasetConfig.to_yaml),
            # so anchor to the yaml's directory rather than the (cwd-relative) save_dir field.
            self.save_dir = os.path.dirname(os.path.abspath(config))
        else:
            self.save_dir = self.config.get("save_dir")
        self.rank = rank
        self.world_size = world_size
        self.shards = self.config.get("shards")
        self._save_label = ""

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

    def copy(self) -> "IMDataset":
        new_dataset = IMDataset(
            config=self.config._config, rank=self.rank, world_size=self.world_size, processing="skip"
        )
        new_dataset.save_dir = self.save_dir
        new_dataset.shard_ids = self.shard_ids.copy()
        new_dataset.data_list = self.data_list.copy()
        return new_dataset

    def __len__(self):
        return len(self.data_list)

    @property
    def processed_names(self) -> list[str]:
        return [
            os.path.join(self.save_dir, f"processed_data_shard{shard_id}{self._save_label}.pkl")
            for shard_id in self.shard_ids
        ]

    def check_exist(self):
        assert all([os.path.exists(name) for name in self.processed_names]), [name for name in self.processed_names]

    def __getitem__(self, index: Union[int, slice]) -> Union[T, list[T]]:
        if isinstance(index, int):
            return self.data_list[index]
        elif isinstance(index, slice):
            ret = self.copy()
            ret.data_list = ret.data_list[index]
            ret.set_index()
            return ret
        else:
            raise TypeError(f"index of type {type(index)} is not allowed.")

    def shuffle(self):
        ret = self.copy()
        random.shuffle(ret.data_list)
        ret.set_index()
        return ret

    def set_index(self):
        for i, data in enumerate(self.data_list):
            data["data_idx"] = i

    def update_data(
        self,
        batched_data: torch.Tensor,
        counts: torch.LongTensor,
        begin_index: int,
        attribute_name: str,
    ):
        idx = 0
        batched_data = batched_data.to("cpu")
        for i, num in enumerate(counts):
            sliced_data = batched_data[idx : idx + num].clone()
            self.data_list[begin_index + i][attribute_name] = sliced_data.clone()
            idx += num

    def load(self, save_dir: str = None, label: str = ""):
        self._save_label = label
        config_save_dir = self.save_dir
        if save_dir is not None:
            self.save_dir = save_dir
        self.data_list = []
        data_cls: Data = getattr(bdata, self.config.get("data_cls"))
        for fname in self.processed_names:
            if not os.path.exists(fname):
                self.save_dir = config_save_dir
                return 0
            self.data_list += [data_cls.from_dict(d) for d in torch.load(fname, weights_only=True)]
        self.set_index()
        self.save_dir = config_save_dir
        return len(self.data_list)

    def save(self, save_dir: str = None, label: str = ""):
        self._save_label = label
        config_save_dir = self.save_dir
        if save_dir is not None:
            self.save_dir = save_dir
        for idx in range(len(self.shard_ids)):
            os.makedirs(os.path.dirname(self.processed_names[idx]), exist_ok=True)
            torch.save([dict(d) for d in self.data_list[idx :: len(self.shard_ids)]], self.processed_names[idx])
        self.save_dir = config_save_dir

    @classmethod
    def process(cls, config: Union[str, dict], shard_id: int, processor: DatasetProcessor = None):
        if processor is None:
            processor = ByteFFDatasetProcessor()

        ds = cls(config, processing=True)
        if os.path.exists(ds.processed_names[shard_id]):
            logger.info(f"shard {shard_id} already processed, skip!")
            return

        logger.info(f"processing shard {shard_id}")
        data_list = []
        shard_inputs = processor.get_shard_inputs(ds, shard_id)

        try:
            for item in shard_inputs:
                if len(data_list) % 1000 == 0:
                    logger.info(f"finished mol {len(data_list)}")

                try:
                    data = processor.process_input(ds, item)
                except Exception:  # pylint: disable=broad-except
                    logger.warning(f"failed: {processor.describe_item(item)}, skip!")
                    logger.warning(traceback.format_exc())
                    continue
                if data is not None:
                    data_list.append(data)
        finally:
            processor.finalize(ds)

        os.makedirs(os.path.dirname(ds.processed_names[shard_id]), exist_ok=True)
        torch.save([dict(d) for d in data_list], ds.processed_names[shard_id])
        # Anchor dataset_config.yaml to the same directory as the shards (ds.save_dir),
        # rather than the config's (cwd-relative) save_dir field, so config and data stay co-located.
        ds.config.to_yaml(save_path=os.path.join(ds.save_dir, "dataset_config.yaml"))

        logger.info(f"finished shard {shard_id}")
