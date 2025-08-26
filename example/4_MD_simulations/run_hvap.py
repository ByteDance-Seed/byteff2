# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import numpy as np
import openmm.app as app
import pandas as pd

from byteff2.md_utils.md_run import npt_run, nvt_run
from byteff2.toolkit.openmmtool import build_system
from bytemol.utils import setup_default_logging

logger = setup_default_logging()

TEMPERATURE = 298
NPT_STEPS = 1500000
NVT_STEPS = 1500000

INPUT_DIR = os.path.abspath("./input_data_hvap")
OUTPUT_DIR = os.path.abspath("./hvap_results")


def post_process(work_dir, nmols, temperature):  # pylint: disable=W0621
    csv_file = os.path.join(work_dir, 'npt_state.csv')
    df = pd.read_csv(csv_file)
    density = df["Density (g/mL)"]
    dd = []
    for _ in range(10):
        dd.append(np.mean(np.random.choice(density[2000:3000], 100)))
    density, density_std = np.mean(dd), np.std(dd)

    e_liquid = df["Potential Energy (kJ/mole)"]
    el = []
    for _ in range(10):
        el.append(np.mean(np.random.choice(e_liquid[2000:3000], 100)) / nmols)
    e_liquid, e_liquid_std = np.mean(el), np.std(el)

    csv_file = os.path.join(work_dir, 'nvt_state.csv')
    df = pd.read_csv(csv_file)
    e_gas = df["Potential Energy (kJ/mole)"]
    eg = []
    for _ in range(10):
        eg.append(np.mean(np.random.choice(e_gas[2000:3000], 100)))
    e_gas, e_gas_std = np.mean(eg), np.std(eg)

    hvap = (e_gas - e_liquid) / 4.184 + 8.314 * temperature / 1000 / 4.184  # kcal/mol
    hvap_std = np.sqrt(e_gas_std**2 + e_liquid_std**2) / 4.184

    result = {
        "density": density,
        "density_std": density_std,
        "hvap": hvap,
        "hvap_std": hvap_std,
    }

    with open(os.path.join(work_dir, 'hvap_results.json'), 'w') as f:
        json.dump(result, f, indent=4)
    logger.info(result)
    return result


if __name__ == '__main__':

    temperature = 298.15
    nmols = 500

    logger.info('building system')
    nonbonded_params = {}
    for mol in ['ACT']:
        nonbonded_params[mol] = json.load(open(f"{INPUT_DIR}/{mol}/{mol}.json", "r"))
    nonbonded_params['metadata'] = {
        "exp6": True,
        "s12": 0.1690015023721859,
        "disp_damping": 120.0,
        "thole": 0.39,
    }
    liq_gro_file = f"{INPUT_DIR}/liquid.gro"
    liq_top_file = f"{INPUT_DIR}/liquid.top"

    grofileparser = app.GromacsGroFile(liq_gro_file)
    input_positions = grofileparser.positions
    unit_cell = grofileparser.getUnitCellDimensions()
    liq_top, liq_system = build_system(liq_top_file, nonbonded_params, unit_cell)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info('running liquid phase')
    npt_run(
        top=liq_top,
        system=liq_system,
        positions=input_positions,
        temperature=TEMPERATURE,
        npt_steps=NPT_STEPS,
        work_dir=OUTPUT_DIR,
    )

    gas_gro_file = f"{INPUT_DIR}/gas.gro"
    gas_top_file = f"{INPUT_DIR}/gas.top"
    grofileparser = app.GromacsGroFile(gas_gro_file)
    input_positions = grofileparser.positions
    gas_top, gas_system = build_system(gas_top_file, nonbonded_params, unit_cell=None)

    logger.info('running gas phase')
    nvt_run(top=gas_top,
            system=gas_system,
            positions=input_positions,
            box_vec=None,
            temperature=TEMPERATURE,
            nvt_steps=NVT_STEPS,
            work_dir=OUTPUT_DIR,
            timestep=0.2)

    post_process(OUTPUT_DIR, nmols, temperature)
