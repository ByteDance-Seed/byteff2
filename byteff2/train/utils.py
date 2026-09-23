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

import logging
import os
from pathlib import Path

import torch
import yaml

from byteff2.bytemol.core import Molecule
from byteff2.data import GraphData
from byteff2.model import EnsembleModel, HybridFF
from byteff2.toolkit import ffparams_to_tfs
from byteff2.toolkit.xtb_calculator import patch_hypervalent_bonded_params_for_write, xtb_optimized_coords
from byteff2.utils import get_asset_path


logger = logging.getLogger(__name__)


def load_training_config(config_path: str, asset_root: str | os.PathLike | None = None) -> dict:
    """Load an example training config and resolve paths independently of the current directory."""
    path = Path(config_path).resolve()
    with path.open() as file:
        config = yaml.safe_load(file)
    config_dir = path.parent

    meta = config.get("meta", {})
    if meta.get("work_folder"):
        meta["work_folder"] = str((config_dir / meta["work_folder"]).resolve())
    for dataset in config.get("dataset", []):
        if dataset.get("config"):
            dataset["config"] = str((config_dir / dataset["config"]).resolve())
    model = config.get("model", {})
    checkpoint_asset = model.pop("check_point_asset", None)
    if checkpoint_asset:
        model["check_point"] = get_asset_path(checkpoint_asset, asset_root)
    elif model.get("check_point"):
        model["check_point"] = str((config_dir / model["check_point"]).resolve())
    return config


def _predict_ffparams(model: HybridFF, data: GraphData):
    device = next(model.parameters()).device
    data = data.to(device)
    with torch.no_grad():
        preds = model(data, skip_ff=True, do_patch=False)
    return data, preds["ff_parameters"]


def _get_hypervalent_patched_ffparams(model: HybridFF, mol: Molecule):
    coords = xtb_optimized_coords(mol)
    mol.conformers[0].coords = coords
    confdata = {"coords": coords[None, ...]}  # numpy [n_conf, n_node, 3]
    data = GraphData("test", mol.get_mapped_smiles(), record_nonbonded_all=False, confdata=confdata, max_n_confs=1)
    data, ffparams = _predict_ffparams(model, data)
    patch_hypervalent_bonded_params_for_write(ffparams, data)
    return data, ffparams


def load_model(
    model_config,
    ckpt: str = None,
    device: str = "cpu",
    strict: bool = True,
) -> HybridFF:

    if isinstance(model_config, str):
        config_path = os.path.join(model_config, "fftrainer_config_in_use.yaml")
        with open(config_path) as file:
            config = yaml.safe_load(file)
        config = config["model"]
        config.pop("check_point", None)
        model = HybridFF(**config).to(device)

        sd = torch.load(
            os.path.join(model_config, "optimal.pt"),
            weights_only=True,
            map_location=device,
        )
        model.load_state_dict(sd["model_state_dict"], strict=strict)

    else:
        assert isinstance(model_config, dict)
        model = HybridFF(**model_config)

    if ckpt is not None:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd["model_state_dict"])
        # model.load_state_dict(sd['model_state_dict'], strict=False)

    model.eval()
    return model


def load_ensemble_model(model_config_list: list[str], ckpt: str = None) -> EnsembleModel:
    models = [load_model(model_config) for model_config in model_config_list]
    print(len(models))
    return EnsembleModel(models)


def load_pretrained_model(
    name: str,
    *,
    asset_root: str | os.PathLike | None = None,
    device: str = "cpu",
    member: int | None = None,
) -> HybridFF | EnsembleModel:
    """Load an official ByteFF2 release model from an external asset root."""
    if name == "ByteFF-26":
        if member is not None and (not isinstance(member, int) or isinstance(member, bool) or not 1 <= member <= 5):
            raise ValueError("ByteFF-26 member must be an integer from 1 through 5")
        members = [member] if member is not None else list(range(1, 6))
        model_dirs = [get_asset_path(f"ByteFF-26/models/member-{index:02d}", asset_root) for index in members]
        if member is not None:
            return load_model(model_dirs[0], device=device).eval()
        model = EnsembleModel([load_model(path, device=device) for path in model_dirs])
        model.eval()
        return model

    if name == "ByteFF-Pol-25":
        if member is not None:
            raise ValueError("ByteFF-Pol-25 does not accept a member")
        model_dir = get_asset_path("ByteFF-Pol-25/model", asset_root)
        return load_model(model_dir, device=device).eval()

    raise ValueError(f"Unknown pretrained model: {name}. Supported models: ByteFF-26, ByteFF-Pol-25")


def get_nb_params(model: HybridFF, mol: Molecule, write_to_itp=False):

    from byteff2.bytemol.core.rkutil.conformer import find_hypervalent_centers

    model.eval()

    # Patch (xtb gfn2 opt for bond/angle at hypervalent centers) is required
    # iff the molecule actually contains a hypervalent center.
    do_patch = bool(find_hypervalent_centers(mol.rkmol))

    metadata = {
        "exp6": True,
        "s12": model.ff_block.ff_layers["Exp6Pol"].s12 / 10.0 * 4.184 ** (1 / 12),  # in kcal^(1/12) * nm
        "disp_damping": model.ff_block.ff_layers["Exp6Pol"].disp_damping_factor,
        "thole": model.ff_block.ff_layers["Exp6Pol"].dipole_solver.a,
    }
    if do_patch:
        data, ffparams = _get_hypervalent_patched_ffparams(model, mol)
    else:
        data = GraphData("test", mol.get_mapped_smiles(), record_nonbonded_all=False, confdata=None, max_n_confs=1)
        data, ffparams = _predict_ffparams(model, data)

    c6 = ffparams["PreExp6Pol.c6"]
    r0 = ffparams["PreExp6Pol.rvdw"]
    lamb = ffparams["PreExp6Pol.lambda"]
    eps = ffparams["PreExp6Pol.eps"]
    lamb = lamb.reshape(-1).detach().tolist()
    eps = (eps * 4.184).reshape(-1).detach().tolist()

    params = {
        "charge": ffparams["PreChargeVolume.charges"].flatten().detach().tolist(),
        "alpha": (ffparams["PreExp6Pol.alpha"] * 1e-3).flatten().detach().tolist(),  # nm^3
        "pol_damping": (ffparams["PreExp6Pol.pol_damping"] * 1e-3).flatten().detach().tolist(),  # nm^3
        "lamb": lamb,
        "eps": eps,  # kJ/mol
        "Rvdw": (r0 * 0.1).flatten().detach().tolist(),  # nm
        "C6": (c6 * 4.184 * 1e-6).flatten().detach().tolist(),  # kJ/mol * nm^6
        "ct_eps": (ffparams["PreExp6Pol.ct_eps"] * 4.184 * 1e-4).flatten().detach().tolist(),  # kj/mol * nm^4
        "ct_lamb": ffparams["PreExp6Pol.ct_lamb"].flatten().detach().tolist(),
    }

    # add fake nonbonded params to itp
    ffparams["PreLJEs.sigma"] = torch.zeros(len(params["charge"]))
    ffparams["PreLJEs.epsilon"] = torch.zeros(len(params["charge"]))
    ffparams["PreLJEs.charges"] = ffparams["PreChargeVolume.charges"].flatten().detach()
    if write_to_itp:
        tfs = ffparams_to_tfs(ffparams, data, mol, mol_name=mol.name)
    else:
        tfs = None

    return metadata, params, tfs, mol
