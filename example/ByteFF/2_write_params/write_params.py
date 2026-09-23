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


"""Generate a single GROMACS ``.itp`` from a mapped SMILES and trained model(s).

A minimal single-molecule counterpart to ``ff_eval/step2_write_itp.py``: give it
one mapped SMILES and one or more trained ByteFF model directories
(containing ``optimal.pt`` and ``fftrainer_config_in_use.yaml``), and it writes
one ``.itp``. Passing several model directories writes ensemble-averaged
parameters.

This model is a standard GNN force field (``MMBonded`` + ``LJEs``), so we call the
low-level ``ffparams_to_tfs`` writer directly.

    python scripts/ByteFF/gen_itp.py \
        --smiles "[H:3][C:1]([H:4])([H:5])[O:2][H:6]" \
        --model /path/to/model_dir_a /path/to/model_dir_b \
        --output methanol.itp
"""

import argparse
import os

import torch

from byteff2.bytemol.core import Molecule
from byteff2.bytemol.utils import setup_default_logging
from byteff2.data import GraphData
from byteff2.model import EnsembleModel
from byteff2.toolkit.gmxtool import ffparams_to_tfs
from byteff2.train.utils import load_model, load_pretrained_model


logger = setup_default_logging()


def _model_device(model):
    """Device of a HybridFF, or of the first member of an EnsembleModel."""
    if isinstance(model, EnsembleModel):
        return next(model.models[0].parameters()).device
    return next(model.parameters()).device


def load_model_or_ensemble(model_dirs: list[str] | None, device: str, asset_root: str | None = None):
    """Load one ``HybridFF`` or an ``EnsembleModel`` from model directories."""
    if not model_dirs:
        return load_pretrained_model("ByteFF-26", asset_root=asset_root, device=device)
    if len(model_dirs) == 1:
        return load_model(model_dirs[0], device=device).eval()
    members = [load_model(d, device=device) for d in model_dirs]
    model = EnsembleModel(members)
    model.eval()
    return model


def predict_ffparams(model, mapped_smiles: str):
    """Run graph_block + preff_block only (skip_ff) to get bonded/nonbonded params."""
    data = GraphData("test", mapped_smiles, record_nonbonded_all=False, confdata=None, max_n_confs=1)
    device = _model_device(model)
    data = data.to(device)
    with torch.no_grad():
        preds = model(data, skip_ff=True, do_patch=False)
    return data, preds["ff_parameters"]


def write_itp(model, mapped_smiles: str, itp_path: str, mol_name: str):
    os.makedirs(os.path.dirname(os.path.abspath(itp_path)) or ".", exist_ok=True)
    mol = Molecule.from_mapped_smiles(mapped_smiles, nconfs=0)
    mol.name = mol_name
    data, ffparams = predict_ffparams(model, mapped_smiles)
    tfs = ffparams_to_tfs(ffparams, data, mol, mol_name=mol_name)
    # atomic write: a crash mid-write leaves a .tmp (ignored) rather than a truncated .itp
    tmp_path = f"{itp_path}.tmp.{os.getpid()}"
    tfs.write_itp(tmp_path, separated_atp=False)
    os.replace(tmp_path, itp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smiles", required=True, help="mapped SMILES of the molecule")
    parser.add_argument(
        "--model",
        nargs="+",
        help="custom trained model dirs; omitted to use the official five-member ByteFF-26 ensemble",
    )
    parser.add_argument("--asset-root", help="ByteFF2 assets root; overrides BYTEFF2_ASSET_ROOT")
    parser.add_argument("--output", required=True, help="output .itp path")
    parser.add_argument("--mol_name", default="MOL", help="moleculetype name written into the itp (default: MOL)")
    parser.add_argument("--device", default="auto", help="'auto' (cuda if available else cpu), or 'cpu'/'cuda'")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("using device: %s", device)

    model_dirs = [os.path.abspath(d) for d in args.model] if args.model else None
    for model_dir in model_dirs or []:
        assert os.path.exists(os.path.join(model_dir, "optimal.pt")), f"optimal.pt not found in {model_dir}"
    if model_dirs and len(model_dirs) == 1:
        logger.info("single model: %s", model_dirs[0])
    elif model_dirs:
        logger.info("ensemble of %d models: %s", len(model_dirs), model_dirs)
    else:
        logger.info("official ByteFF-26 five-member ensemble")
    model = load_model_or_ensemble(model_dirs, device=device, asset_root=args.asset_root)

    write_itp(model, args.smiles, args.output, args.mol_name)
    logger.info("wrote itp for %s -> %s", args.mol_name, args.output)


if __name__ == "__main__":
    main()
