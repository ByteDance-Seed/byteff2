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

from byteff2.md_utils.md_run import npt_run
from byteff2.toolkit.openmmtool import build_system
from bytemol.utils import setup_default_logging

logger = setup_default_logging()

INPUT_DIR = os.path.abspath("./input_data")
OUTPUT_DIR = os.path.abspath("./density_results")


def post_process(post_work_dir):
    csv_file = os.path.join(post_work_dir, 'npt_state.csv')
    density = pd.read_csv(csv_file)["Density (g/mL)"]

    dd = []
    for _ in range(10):
        dd.append(np.mean(np.random.choice(density[2000:3000], 100)))
    density, density_std = np.mean(dd), np.std(dd)
    result = {
        "density": density,
        "density_std": density_std,
    }
    with open(os.path.join(post_work_dir, 'density_results.json'), 'w') as f:
        json.dump(result, f, indent=4)
    logger.info(result)
    return result


if __name__ == '__main__':

    logger.info('building system')
    nonbonded_params = {}
    for mol in [
            'DMC',
            'EC',
            'LI',
            'PF6',
    ]:
        nonbonded_params[mol] = json.load(open(f"{INPUT_DIR}/{mol}/{mol}.json", "r"))
    nonbonded_params['metadata'] = {
        "exp6": True,
        "s12": 0.1690015023721859,
        "disp_damping": 120.0,
        "thole": 0.39,
    }
    gro_file = f"{INPUT_DIR}/solvent_salt.gro"
    top_file = f"{INPUT_DIR}/topol.top"

    grofileparser = app.GromacsGroFile(gro_file)
    input_positions = grofileparser.positions
    unit_cell = grofileparser.getUnitCellDimensions()

    input_top, input_system = build_system(top_file, nonbonded_params, unit_cell)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npt_run(
        top=input_top,
        system=input_system,
        positions=input_positions,
        temperature=298,
        npt_steps=1500000,
        work_dir=OUTPUT_DIR,
    )
    post_process(OUTPUT_DIR)
