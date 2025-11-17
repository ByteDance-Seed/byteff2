# Data Preparation Example

This directory contains example scripts for preparing molecular data for training `ByteFF-Pol`.

## Usage

### 1. Generate Conformers

To generate optimized molecular conformers:

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python generate_conf.py --mol_name ACT --mapped_smiles "[C:1]([C:2](=[O:3])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]"
```

This script uses geomeTRIC optimizer with B3LYP/def2-SVPD to optimize the molecular structure and saves it as an XYZ file.

### 2. Generate Dimers

To create molecular dimer configurations:

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python generate_dimer.py --mol1 ACT.xyz --mol2 EC.xyz --save_dir ACT_EC_dimer --nconfs 100
```

This script using optimized structure of step 1 to generates multiple dimer configurations by randomly rotating and translating molecules while maintaining minimum/maximum distance constraints.

### 3. Calculate EDA

To perform Energy Decomposition Analysis on dimers:

```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python calculate_eda.py --input_dir ACT_EC_dimer/conf_0
```

This script performs ALMO-EDA calculations using wB97M-V/def2-TZVPD to extract interaction energy components (electrostatic, exchange, polarization, charge transfer, etc.).
