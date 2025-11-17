# QM vs FF Comparison Examples

This example demonstrates how to compare quantum mechanical (QM) and force field (FF) energies for molecular dimers and clusters using `ByteFF-Pol`.
The scripts reproduce the results presented in Figure 2 (a~f) of the ByteFF-Pol paper.

## Usage
1. Dimer validation examples:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python dimer_validation.py
```
The results shown in Figure 2 (a) is reproduced (Fig2a.png).
One can change the device to `cuda` for faster computation.

2. Dimer scan examples:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python compare_dimer.py
```
Results will be saved in the `dimer_results` folder, which contains JSON files with decomposed interaction energies (predicted by ByteFF-Pol and QM references) and corresponding visualization images.

3. Cluster optimization examples:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python compare_cluster.py
```
Results will be saved in the `cluster_results` folder, which contains geometry optimization trajectories for each cluster and energy comparison results in `results.csv`.

In the CSV file, each decomposed energy column contains three values:
- The first value (in parentheses) is the QM reference energy
- The second value is the ByteFF-Pol energy evaluated on the QM geometry
- The third value is the ByteFF-Pol energy evaluated on the FF-relaxed geometry

Energy values are in kcal/mol, and RMSD values are in $\mathrm{\AA}$.