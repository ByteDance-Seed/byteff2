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
import logging
import os
import random

import h5py
from torch import Tensor

from byteff2.bytemol.utils import get_data_file_path
from byteff2.data import ClusterData, Data, DatasetProcessor, IMDataset, MonoData
from byteff2.data.data import HessianData
from byteff2.utils.definitions import MMTerm, MMTERM_WIDTH


logger = logging.getLogger(__name__)

mono_meta_fp = get_data_file_path("monomer_100/meta.txt", "byteff2.tests.testdata")
mono_json_fp = get_data_file_path("monomer_100/monomer_100.json", "byteff2.tests.testdata")
mono_hdf5_fp = get_data_file_path("monomer_100/monomer_100.h5", "byteff2.tests.testdata")

dimer_meta_fp = get_data_file_path("dimer_100/meta.txt", "byteff2.tests.testdata")
dimer_json_fp = get_data_file_path("dimer_100/dimer100.json", "byteff2.tests.testdata")
dimer_hdf5_fp = get_data_file_path("dimer_100/dimer100.h5", "byteff2.tests.testdata")

hessian_meta_fp = get_data_file_path("hessian_data/meta.txt", "byteff2.tests.testdata")


def create_mono_dataset(path, max_n_confs=10):

    config = {
        "meta_fp": mono_meta_fp,
        "save_dir": path,
        "data_cls": "MonoData",
        "confdata": {
            "coords": "coords",
            "energy": "energy",
            "forces": "forces",
        },
        "kwargs": {"max_n_confs": max_n_confs},
    }

    # processing and saving
    IMDataset.process(config, 0)
    dataset = IMDataset(config)
    return config, dataset


def test_dataset_monodata(tmp_path):

    max_n_confs = 16
    config, dataset = create_mono_dataset(tmp_path, max_n_confs)

    idx = random.randint(1, len(dataset) - 1)
    with open(mono_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(",")

    h5 = h5py.File(mono_hdf5_fp)
    h5_dataset = h5[name]
    with open(mono_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = MonoData(
        name,
        smiles,
        confdata=dict(coords=h5_dataset["coords"][:], energy=h5_dataset["energy"][:], forces=h5_dataset["forces"][:]),
        max_n_confs=max_n_confs,
    )

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all()
        else:
            print(k, v)

    h5.close()

    # test loading
    dataset = IMDataset(config, processing=False)

    idx = random.randint(1, len(dataset) - 1)
    with open(mono_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(",")

    h5 = h5py.File(mono_hdf5_fp)
    h5_dataset = h5[name]
    with open(mono_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = MonoData(
        name,
        smiles,
        confdata=dict(coords=h5_dataset["coords"][:], energy=h5_dataset["energy"][:], forces=h5_dataset["forces"][:]),
        max_n_confs=max_n_confs,
    )

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all()
        else:
            raise TypeError(f"Unknow type of {k}: {type(v)}")

    h5.close()


def create_hessian_dataset(max_n_confs=10):
    config = {
        "confdata": {
            "coords": "coords",
            "hessian": "hessian",
        },
        "data_cls": "HessianData",
        "kwargs": {
            "max_n_confs": 20,
        },
        "meta_fp": hessian_meta_fp,
        "moldata": {
            "abcg2_charges": "abcg2_charges",
            "angle_d0": "angle_theta",
            "angle_k": "angle_k",
            "bond_k": "bond_k",
            "bond_r0": "bond_length",
            "epsilon": "epsilon",
            "improper_k": "impropertorsion_k",
            "proper_k": "propertorsion_k",
            "sigma": "sigma",
        },
        "save_dir": os.path.dirname(hessian_meta_fp),
        "shards": 32,
    }
    dataset = IMDataset(config, shard_id=0, processing=False)
    return dataset


def test_dataset_hessiandata(tmp_path):
    dataset = create_hessian_dataset()

    # dataset must be non-empty so downstream tests get real samples
    assert len(dataset) > 0

    for data in dataset:
        # type and basic identity
        assert isinstance(data, HessianData)
        assert isinstance(data["mol_name"], str) and data["mol_name"]
        assert isinstance(data["mapped_smiles"], str) and data["mapped_smiles"]

        # partial_hessian shape contract: [nPartialHessian, nconfs, 9]
        ph = data["partial_hessian"]
        assert isinstance(ph, Tensor)
        assert ph.dim() == 3 and ph.shape[-1] == 9
        assert ph.shape[0] == data.get_count("partial_hessian")
        assert ph.shape[1] == data["coords"].shape[1]  # nconfs

        # bonded/nonbonded GAFF2 labels are loaded with matching counts
        assert data["bond_k"].shape[0] == data.get_count("bond")
        assert data["bond_r0"].shape[0] == data.get_count("bond")
        assert data["angle_k"].shape[0] == data.get_count("angle")
        assert data["angle_d0"].shape[0] == data.get_count("angle")
        assert data["proper_k"].shape[0] == data.get_count("proper")
        assert data["improper_k"].shape[0] == data.get_count("improper")
        natoms = data.get_count("node")
        assert data["abcg2_charges"].shape[0] == natoms
        assert data["sigma"].shape[0] == natoms
        assert data["epsilon"].shape[0] == natoms

        # each MMTerm has a `_rec_i_j` index tensor whose length matches its term count
        for term in [MMTerm.bond, MMTerm.angle, MMTerm.proper, MMTerm.improper]:
            width = MMTERM_WIDTH[term]
            term_count = data.get_count(term.name)
            for i in range(width):
                for j in range(width):
                    if i == j:
                        continue
                    if abs(i - j) >= 3 and term is MMTerm.proper:
                        continue
                    key = f"{term.name}_rec_{i}_{j}"
                    assert key in data
                    rec = data[key]
                    assert rec.dim() == 1 and rec.shape[0] == term_count
                    # rec must index into partial_hessian rows
                    if term_count > 0:
                        assert int(rec.max()) < ph.shape[0]
                        assert int(rec.min()) >= 0


def create_dimer_dataset(path, max_n_confs=10):

    config = {
        "meta_fp": dimer_meta_fp,
        "save_dir": path,
        "data_cls": "ClusterData",
        "confdata": {
            "coords": "coords",
            "forces_cluster": "forces_cluster",
            "energy_cluster": "energy_cluster",
        },
        "kwargs": {"max_n_confs": max_n_confs},
    }

    # processing and saving
    IMDataset.process(config, 0)
    dataset = IMDataset(config)
    return config, dataset


def test_dataset_dimer(tmp_path):

    max_n_confs = 50
    config, dataset = create_dimer_dataset(tmp_path, max_n_confs)

    idx = random.randint(1, len(dataset) - 1)
    with open(dimer_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(",")

    h5 = h5py.File(dimer_hdf5_fp)
    h5_dataset = h5[name]
    with open(dimer_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = ClusterData(
        name,
        smiles,
        confdata=dict(
            coords=h5_dataset["coords"][:],
            forces_cluster=h5_dataset["forces_cluster"][:],
            energy_cluster=h5_dataset["energy_cluster"][:],
        ),
        max_n_confs=max_n_confs,
    )

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all(), k
        else:
            assert v == data[k], k

    h5.close()

    # test loading
    dataset = IMDataset(config, processing=False)

    idx = random.randint(1, len(dataset) - 1)
    with open(dimer_meta_fp) as file:
        lines = file.readlines()
        _, name = lines[idx].rstrip().split(",")

    h5 = h5py.File(dimer_hdf5_fp)
    h5_dataset = h5[name]
    with open(dimer_json_fp) as file:
        smiles = json.load(file)[name]

    tdata = ClusterData(
        name,
        smiles,
        confdata=dict(
            coords=h5_dataset["coords"][:],
            forces_cluster=h5_dataset["forces_cluster"][:],
            energy_cluster=h5_dataset["energy_cluster"][:],
        ),
        max_n_confs=max_n_confs,
    )

    data = dataset[idx]
    for k, v in tdata.items():
        assert k in data
        if isinstance(v, str):
            assert v == data[k]
        elif isinstance(v, Tensor):
            assert (v == data[k]).all()
        else:
            assert v == data[k], k

    h5.close()


class _SkipOnProcessErrorProcessor(DatasetProcessor):
    def iter_inputs(self, ds: IMDataset):
        del ds
        yield {"name": "ok-1"}
        yield {"name": "bad"}
        yield {"name": "ok-2"}

    def process_input(self, ds: IMDataset, item: dict):
        if item["name"] == "bad":
            raise RuntimeError("bad sample")
        return Data.from_dict({"name": item["name"]})


def test_imdataset_process_skips_bad_sample(tmp_path):
    config = {
        "meta_fp": mono_meta_fp,
        "save_dir": str(tmp_path),
        "data_cls": "Data",
    }

    IMDataset.process(config, 0, processor=_SkipOnProcessErrorProcessor())
    dataset = IMDataset(config)

    assert [data["name"] for data in dataset] == ["ok-1", "ok-2"]


def test_byteff_dataset_processor_skips_missing_csv_semantics_and_bad_h5(caplog, tmp_path):
    meta_fp = tmp_path / "meta.csv"
    h5_fp = tmp_path / "data.h5"
    save_dir = tmp_path / "processed"

    with open(mono_json_fp) as file:
        valid_mapped_smiles = next(iter(json.load(file).values()))

    meta_fp.write_text(
        "uuid,mapped_nonisomeric_smiles,mapped_isomeric_smiles,h5_file\n"
        "missing-smiles,C,nan,data.h5\n"
        f"missing-group,C,{valid_mapped_smiles},data.h5\n"
        f"good,C,{valid_mapped_smiles},data.h5\n",
        encoding="utf-8",
    )

    with h5py.File(h5_fp, "w") as h5_file:
        group = h5_file.create_group("good")
        group.create_dataset("coords", data=[[[0.0, 0.0, 0.0]]])
        group.create_dataset("energy", data=[0.0])
        group.create_dataset("forces", data=[[[0.0, 0.0, 0.0]]])

    config = {
        "meta_fp": str(meta_fp),
        "save_dir": str(save_dir),
        "data_cls": "MonoData",
        "confdata": {
            "coords": "coords",
            "energy": "energy",
            "forces": "forces",
        },
        "kwargs": {"max_n_confs": 4},
    }

    with caplog.at_level(logging.WARNING):
        IMDataset.process(config, 0)

    dataset = IMDataset(config)
    assert len(dataset) == 1
    assert dataset[0]["mol_name"] == "good"
    assert dataset[0]["mapped_smiles"] == valid_mapped_smiles
    assert "failed: missing-smiles, skip!" in caplog.text
    assert "failed: missing-group, skip!" in caplog.text
