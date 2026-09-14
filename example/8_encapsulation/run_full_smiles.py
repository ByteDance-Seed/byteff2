#!/usr/bin/env python
"""Run all by SMILES

FTA ( asymetric version of FSA or TFSA anion) is not in the ByteFF2 inventory, so it is declared through custom_smiles.
custom_smiles covers solvents and anions alike; here both styles are mixed in
one system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from byteff2.toolkit.common import PROPERTIES, save
from byteff2.toolkit.properties_calculator import PropertiesCalculator

BASE_DIR = "./run_smiles_EC_DMC"

calc = PropertiesCalculator(
    solvent=["EC", "DMC"],
    solvent_ratio="5:5",
    anion="FTA",
    li_count=34,
    salt_to_solvent_ratio_str="1:10",
    custom_smiles={"FTA": "[N-](S(=O)(=O)F)S(=O)(=O)C(F)(F)F"},
    base_dir=BASE_DIR,
)

if __name__ == "__main__":
    save(calc.calculate(properties=PROPERTIES), BASE_DIR)
