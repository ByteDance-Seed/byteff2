#!/usr/bin/env python
"""Pure-solvent (zero-salt) control for run_full_smiles.py's LiTFA/EC/DMC run.

Same EC:DMC composition and solvent molecule counts as run_full_smiles.py
(170 EC + 170 DMC, T=298K) but with no Li+/FTA- at all. Tests whether that
run's anomalously high dielectric constant (390.4, vs. literature EC~90/
DMC~3) is an artifact of unbounded ionic translational diffusion
contaminating the dipole-fluctuation formula: this solvent-only run should
land in a plausible range if that diagnosis is right.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from byteff2.toolkit.common import save
from byteff2.toolkit.properties_calculator import PropertiesCalculator

BASE_DIR = "./run_solvent_only_EC_DMC"  # new dir name -- avoids the crashed draft's leftover state in ./run_EC_DMC

calc = PropertiesCalculator(
    solvent=["EC", "DMC"],
    solvent_ratio="5:5",
    li_count=0,
    total_solvent=340,  # 170 EC + 170 DMC, matching run_smiles_EC_DMC's solvent counts exactly
    temperature=298.0,
    base_dir=BASE_DIR,
)

if __name__ == "__main__":
    # Conductivity is skipped from the requested/reported properties (meaningless
    # for a neutral system anyway). Note this doesn't skip any compute: "viscosity"
    # alone still runs the same single TransportProtocol/NEMD job that would also
    # produce conductivity_onsager -- it's just not requested/promoted here.
    save(calc.calculate(properties=["density", "viscosity", "dielectric"]), BASE_DIR)
