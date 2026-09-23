# Example 2: Writing Force Field Parameters

This example demonstrates how to generate force field parameters for a molecule
using trained `ByteFF` model(s) and write them to a GROMACS-compatible
`.itp` file.

## Overview

The script loads one or more trained `ByteFF` model directories, predicts
bonded and nonbonded force field parameters from a mapped SMILES string, and
writes a single `.itp` file.

Each model directory must contain:

1. `optimal.pt`
2. `fftrainer_config_in_use.yaml`

With no `--model`, the script loads the official five-member ByteFF-26 ensemble from `BYTEFF2_ASSET_ROOT`.
Passing one custom model directory writes parameters from that model; passing multiple custom directories writes
ensemble-averaged parameters.

## File in this directory

`write_params.py`: Main script that generates and writes force field parameters.

## Usage

### 1. Generate an ITP with the Official Ensemble

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python write_params.py \
    --mol_name AFGBL \
    --smiles '[O:1]=[C:2]1[O:3][C:4]([H:8])([H:9])[C:5]([H:10])([H:11])[C@@:6]1([F:7])[H:12]' \
    --output AFGBL.itp
```

### 2. Generate an ITP with One Custom Model

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python write_params.py \
    --mol_name AFGBL \
    --smiles '[O:1]=[C:2]1[O:3][C:4]([H:8])([H:9])[C:5]([H:10])([H:11])[C@@:6]1([F:7])[H:12]' \
    --model /path/to/byteff-models/model_a \
    --output AFGBL.itp
```

### 3. Generate an ITP with Multiple Custom Models

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python write_params.py \
    --mol_name AFGBL \
    --smiles '[O:1]=[C:2]1[O:3][C:4]([H:8])([H:9])[C:5]([H:10])([H:11])[C@@:6]1([F:7])[H:12]' \
    --model /path/to/byteff-models/model_a /path/to/byteff-models/model_b /path/to/byteff-models/model_c \
    --output AFGBL.itp
```

When several model directories are provided, the script builds an ensemble and
writes averaged force field parameters.

### 4. Show Help

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} \
python write_params.py -h
```

## Arguments

- `--smiles`: mapped SMILES of the molecule.
- `--model`: optional custom trained model directories. Each directory must contain `optimal.pt` and
  `fftrainer_config_in_use.yaml`. If omitted, the official ByteFF-26 ensemble is used.
- `--asset-root`: optional assets root; overrides `BYTEFF2_ASSET_ROOT`.
- `--output`: output `.itp` path.
- `--mol_name`: moleculetype name written into the `.itp`; defaults to `MOL`.
- `--device`: `auto`, `cpu`, or `cuda`; defaults to `auto`.
