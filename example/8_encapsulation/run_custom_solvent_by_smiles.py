#!/usr/bin/env python
"""Solvent by SMILES: 1:10 LiPF6 in DMSO.

DMSO is not in the ByteFF2 inventory, so it is declared through custom_smiles.
ByteFF2 predicts parameters from the molecular graph and will run it, but the
molecule sits outside the training set, so its parameters are extrapolated.

Interface
---------
Inputs  : solvent="DMSO", custom_smiles={"DMSO": "CS(C)=O"}  (by SMILES)
          anion="PF6"                                         (by name)
          li_count=34, salt_to_solvent_ratio_str="1:10"
Output  : ./run_by_smiles/  (results.json + per-property subdirectories)
Properties calculated (PROPERTIES):
    density, viscosity, conductivity
"""

from byteff2.toolkit.common import PROPERTIES, save
from byteff2.toolkit.properties_calculator import PropertiesCalculator

BASE_DIR = "./run_custom_solvent_by_smiles"

calc = PropertiesCalculator(
    solvent="DMSO",
    anion="PF6",
    li_count=34,
    salt_to_solvent_ratio_str="1:10",
    custom_smiles={"DMSO": "CS(C)=O"},
    base_dir=BASE_DIR,
)

if __name__ == "__main__":
    save(calc.calculate(properties=["density", "viscosity", "conductivity"]), BASE_DIR)
