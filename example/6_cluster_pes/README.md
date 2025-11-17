# Example 6: Cluster Potential Energy Surface (PES)
This example demonstrates how to reproduce the cluster PES validation as demonstrated in the SI of ByteFF-Pol paper.

## Overview
In this example, we first perform MD simulations to generate trajectories for sample clusters.
Then, we sample clusters and compute their interaction energies by DFT.
Finally, we compare the interaction energies computed by DFT with those predicted by ByteFF-Pol.

Since the MD simulation and DFT calculations are time-consuming and require GPUs, we provide pre-computed clusters in the `data/md_sample_examples` directory.
You can directly use these results to reproduce the figures in the paper.

## How to Run
0. Set PYTHONPATH
```bash
export PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH}
export OMP_NUM_THREADS=1
```
1. Run MD simulations
If you want to run MD simulations for density calculations, run:
```bash
python3 run_md.py --config data/md_configs/n1n3n4.json
python3 run_md.py --config data/md_configs/n2n5.json
python3 run_md.py --config data/md_configs/i1i4.json
python3 run_md.py --config data/md_configs/i2i5.json
python3 run_md.py --config data/md_configs/i3.json
```
The config files for other simulations, like evaporation enthalpy (Hvap) and transport properties, are also provided. To run these simulations, simply replace `density_config.json` with the corresponding config file.

2. Sample clusters
``` bash
python3 sample_clusters.py --md_samples_dir data/md_samples
```
Sample clusters from the MD trajectories, and save them in the `data/md_samples` directory.

3. Compute DFT interaction energies
``` bash
python3 qm_calc.py --md_samples_dir data/md_samples
```
Compute DFT interaction energies for the sampled clusters saved in the `data/md_samples` directory.

The `gpu4pyscf` package is required to run the DFT calculations. Please refer to the [installation guide](https://github.com/pyscf/gpu4pyscf) for installation.

4. Compare DFT interaction energies with ByteFF-Pol predictions
``` bash
python3 compare_qm_ff.py --md_samples_dir data/md_samples
```
Compare the interaction energies computed by DFT with those predicted by ByteFF-Pol. Results are saved in the `cluster_pes` directory.

The figures in the paper are reproduced by setting `--md_samples_dir data/md_sample_examples`.
