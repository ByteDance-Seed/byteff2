# Training Example

This directory contains example scripts for training `ByteFF-Pol`.

## Overview

The training process involves two main steps:
1. **Data Preprocessing**: Convert raw data into a format suitable for training.
2. **Model Training**: Train the force field model using the preprocessed data.

## Usage

### 1. Data Preprocessing

Before training, you need to preprocess your data:

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python preprocess.py --conf preprocess_example.yaml
```
This script reads the configuration from `preprocess_example.yaml` and processes the data accordingly.

### 2. Model Training

To start training the model, run:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python train.py --conf train.yaml
```
