# Example 4: Molecular Dynamics Simulations
This example demonstrates how to perform molecular dynamics (MD) simulations using the ByteFF-Pol force field with OpenMM.

## Overview
The MD simulations example shows how to:
* Run NPT (constant pressure and temperature) simulations for density calculations
* Run liquid and gas phase simulations to evaluate evaporation enthalpy (Hvap).
* Run simulation for electrolyte and compute viscosity and conductivity.

## How to Run
1. Density Calculation
Run simulation using the parameters from `input_data`
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python run_density.py
```
After simulation, the results are saved in `density_results/density_results.json`.

2. Hvap Calculation
Run simulation using the parameters from `input_data_hvap`
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python run_hvap.py
```
After simulation, the results are saved in `hvap_results/hvap_results.json`.

3. Transport Properties
Run simulations using the parameters from `input_data`
```bash 
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} python run_transport.py
```
After simulation, the results are saved in `transport_results/results.json`.