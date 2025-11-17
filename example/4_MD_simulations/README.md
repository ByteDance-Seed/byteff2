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
