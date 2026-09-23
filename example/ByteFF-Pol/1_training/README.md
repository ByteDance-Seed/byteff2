# Training Example

This directory contains example scripts for training `ByteFF-Pol`.

## Overview

The training process involves two main steps:
1. **Data Preprocessing**: Convert raw data into a format suitable for training.
2. **Model Training**: Train the force field model using the preprocessed data.

## Usage

First, follow [Model Weights and Data](../../../README.md#model-weights-and-data) to materialize the assets and set
`BYTEFF2_ASSET_ROOT`. Then enter this example directory:

```bash
cd example/ByteFF-Pol/1_training
```

### 1. Data Preprocessing

The example configuration reads `ByteFF-Pol-25/data/training/cluster/` through the configured asset root and writes
processed data to this example directory:

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python preprocess.py --conf preprocess_example.yaml
```
This script reads the configuration from `preprocess_example.yaml` and processes the data accordingly.
Pass `--asset-root /path/to/byteff2-assets` to override `BYTEFF2_ASSET_ROOT` for one run. The processed dataset is
written beside this example, not into the assets checkout.

### 2. Model Training

To start training the model, run:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python train.py --conf train.yaml
```

`train.yaml` resolves the official ByteFF-Pol-25 warm-start checkpoint through the same asset root and writes logs
beside this example. An explicit `model.check_point` remains supported for custom training configurations.
