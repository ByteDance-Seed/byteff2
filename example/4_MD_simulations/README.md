# Example 4: Molecular Dynamics Simulations
This example demonstrates how to perform molecular dynamics (MD) simulations using the ByteFF-Pol force field with OpenMM.

## Overview
The MD simulations example shows how to:
* Run NPT simulations for density calculations
* Run liquid and gas phase simulations to evaluate evaporation enthalpy (Hvap).
* Conduct a simulation to compute transport properties such as viscosity, conductivity and so on.

## How to Run
0. Set PYTHONPATH
```bash
export PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH}
```
1. Run MD simulations
If you want to run MD simulations for density calculations, run:
```bash
python run_md.py --config density_config.json
```
The config files for other simulations, like evaporation enthalpy (Hvap) and transport properties, are also provided. To run these simulations, simply replace `density_config.json` with the corresponding config file.

## Configuration File Details (*_config.json)

This configuration is used for running transport property simulations (viscosity and conductivity) on electrolyte systems:

* **protocol**: "Transport" - Specifies the simulation protocol type, including `Transport`, `Density` and `HVap`.
* **temperature**: 298 - Simulation temperature in Kelvin
* **natoms**: 10000 - Total number of atoms in the box
* **components**: Molecular composition with **molecule ratio**:
  - **DMC**: 249 
  - **EC**: 170 
  - ...
* **smiles**: SMILES strings for each component.
  - **DMC**: "COC(=O)OC"
  - ...

Optional box controls (for all protocols):
- `box_length` (number, nm): Overrides the initial cubic box edge length used to pack molecules (GROMACS `editconf -box`).
- `box_scale` (number): Multiplier applied to the internally predicted box length; ignored if `box_length` is provided.

Optional composition and timing controls:
- `components_as_counts` (bool): If true, the values in `components` are treated as exact molecule counts. `natoms` is recomputed from those counts to stay consistent.
- `components_counts` (object): Alternatively provide a separate map of exact molecule counts here; if present it takes precedence over `components`.
- `npt_steps`, `nvt_steps`, `nonequ_steps` (integers): Override default MD lengths (steps) for the respective phases.
- `npt_time_ns`/`ps`, `nvt_time_ns`/`ps`, `nonequ_time_ns`/`ps` (numbers): Specify total time instead of steps. Conversion uses 2 fs for NPT/NVT, and 1 fs for the non-equilibrium viscosity run.

Restart controls:
- `resume` (bool): If true and a phase checkpoint exists, resumes NPT/NVT/nonequ runs from the latest checkpoint (continues CSV/DCD/viscosity logs when supported).
- `checkpoint_interval` (int, steps): Frequency to write checkpoints during a run (default 5000).

Files produced per phase for restart:
- NPT: `npt.chk`, `npt_state.csv`, `npt.dcd`
- NVT: `nvt.chk`, `nvt_state.csv`, `nvt.dcd`
- Nonequilibrium: `nonequ.chk`, `viscosity.csv`

Notes:
- Packing still auto-expands the box by 5% if insertion fails (to ensure the requested composition/atom count fits).
- The periodic unit cell used for MD comes from the generated `.gro` box.
- Ionic conductivity and viscosity controls (Transport protocol):
  - `compute_viscosity` (bool, default true): Runs the non-equilibrium segment and computes viscosity from `viscosity.csv`.
  - `compute_conductivity` (bool, default true): Computes ionic conductivity from NVT frames using Onsager theory; results saved to `results.json`.
  - `viscosity_cP` (number, optional): If provided and `compute_viscosity` is false, uses this value for Yeh–Hummer finite-size correction. If not provided, the conductivity calculation proceeds but skips YH correction (flagged in results as `yh_correction_applied: false`).

Notes on conductivity:
- Conductivity is computed from the equilibrium NVT trajectory (`nvt.dcd`). The non-equilibrium run is only used to compute viscosity for Yeh–Hummer correction of diffusivities; Onsager conductivity itself does not require the non-equilibrium frames.

Optional MSD/fit window controls (Transport):
- `msd_skip_frames` (int, default 200): Number of initial NVT frames to discard before MSD/correlation analysis.
- `fit_window_frames` ([start, end], default [50, 200]): Frame-index window used to linearly fit the kMSD/correlation to obtain slopes. Ensures at least two points.
- `fit_window_frac` ([start_frac, end_frac], optional): Fractional window (0–1) of the analyzed frames; if provided, overrides `fit_window_frames`.

Frame-to-time mapping:
- With default settings, NVT saves a frame every 500 steps and timestep is 2 fs, so 1 frame ≈ 1 ps. If you change report intervals or timesteps, adjust the windows accordingly.

Output labels for transport:
- The results now include `species_labels`, `species_charges`, and `species_counts`, ordered consistently with the computed arrays (e.g., `Dself_inf`).
- By default the order is: anions, then cations, then solvents (matching the composition order constructed in the protocol).
- Example:
  - `species_labels`: ["PF6", "LI", "EC", "DMC"]
  - `Dself_inf`: [0.85, 1.10, 2.45, 2.30]  # 10^-10 m^2/s → PF6, LI, EC, DMC respectively

Per-species outputs and transference numbers:
- `Dself_raw` (10^-10 m^2/s): raw per-species self-diffusion (finite-size, before Yeh–Hummer).
- `Dself_inf` (10^-10 m^2/s): Yeh–Hummer corrected per-species self-diffusion (if viscosity available).
- `Lambda_raw`, `Lambda_com_removed` (10^-10 m^2/s): Einstein-form Onsager matrices before/after COM correction.
- `output_transference` (bool, default false): when true, adds:
  - `transference_numbers`: map from species label to transference number t_i (dimensionless) computed from Onsager L_hat.
  - `t_plus_charge_+1` and `t_minus_charge_-1`: convenience keys when exactly one +1 and one −1 species exist.

Notes:
- The transference numbers are computed in the barycentric (mass-average) frame and do not require velocities.
- For multi-cation or multi-anion systems with the same integer charge, species may be grouped unless distinct species are defined in the input topology and config.
