# Training Example

This directory contains example scripts for training `ByteFF`.

## Overview

The training process involves three main steps:
1. **Generate GAFF2 + AM1BCC Parameters**: Use `antechamber` + `acpype` to generate force field parameters from hessian data.
2. **Data Preprocessing**: Convert raw data into a format suitable for training.
3. **Model Training**: Train the force field model using the preprocessed data (pretrain then joint train).

## Usage

First, follow [Model Weights and Data](../../../README.md#model-weights-and-data) to materialize the assets and set
`BYTEFF2_ASSET_ROOT`. Then enter this example directory:

```bash
cd example/ByteFF/1_training
```

The commands below read raw inputs from `ByteFF-26/data/training/` under that asset root. Generated GAFF2 data,
preprocessed datasets, checkpoints, and logs stay in this example directory rather than modifying the asset tree.

### 1. Generate GAFF2 + AM1BCC Parameters

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python gen_gaff2_am1bcc_h5.py \
    --src-csv "$BYTEFF2_ASSET_ROOT/ByteFF-26/data/training/hessian/hessian_example.csv" \
    --src-h5  "$BYTEFF2_ASSET_ROOT/ByteFF-26/data/training/hessian/hessian_example.h5" \
    --out-h5  ./work/gaff2_am1bcc_hessian_example.h5 \
    --out-csv ./work/gaff2_am1bcc_hessian_example.csv
```

Create the output directory first with `mkdir -p work`. `preprocess_gaff2.yaml` reads the generated CSV/HDF5 from
that directory. `preprocess_torsion.yaml` resolves its read-only source through `BYTEFF2_ASSET_ROOT`. Both
preprocessing commands write their processed datasets beside these example files, independent of the current shell
directory used to invoke the scripts.

Install `acpype` (which bundles AmberTools) into the Python environment you use to run this step. See https://github.com/alanwilter/acpype for installation instructions.

### 2. Data Preprocessing

```bash
# Preprocess hessian data
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python preprocess.py --conf preprocess_gaff2.yaml

# Preprocess torsion data
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python preprocess.py --conf preprocess_torsion.yaml
```

### 3. Model Training

```bash
# Pretrain: fit GAFF2 parameters on hessian data
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python train.py --conf pretrain.yaml

# Joint train: fine-tune on hessian + torsion data
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python train.py --conf train.yaml
```

Set `meta.is_joint: true` in the config to use `FFJointTrainer` (multi-dataset with ffopt); otherwise `FFTrainer` is used.
