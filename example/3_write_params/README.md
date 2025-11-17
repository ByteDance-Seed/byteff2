# Example 3: Writing Force Field Parameters
This example demonstrates how to generate force field parameters for molecules using a trained ByteFF-Pol model and write them to GROMACS-compatible files.

## Overview
The script loads a pre-trained ByteFF-Pol model, generates force field parameters for small molecules, and outputs:
* `.itp` files containing GROMACS-compatible force field parameters
* `.gro` files containing molecular coordinates
* `.json` files containing non-bonded parameters
* `_nb_params.json` files containing metadata

## Files in this directory
`write_params.py`: Main script that generates and writes force field parameters
`AFGBL/`: Example directory containing expected output files for comparison

## Usage
Run the example with:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python write_params.py --mol_name AFGBL --mapped_smiles '[O:1]=[C:2]1[O:3][C:4]([H:8])([H:9])[C:5]([H:10])([H:11])[C@@:6]1([F:7])[H:12]' --out_dir ./AFGBL
```
This will generate force field parameters for AFGBL and write them to the specified output directory.