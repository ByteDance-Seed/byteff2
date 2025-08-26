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

import openmm.app as app

from byteff2.md_utils.md_run import dcd_read, npt_run, nvt_run, rescale_box, volume_calc
from byteff2.md_utils.onsager_conductivity import onsager_calc
from byteff2.md_utils.viscosity import nonequ_run, viscosity_calc
from byteff2.toolkit.openmmtool import build_system
from bytemol.utils import setup_default_logging, temporary_cd

logger = setup_default_logging()

TEMPERATURE = 298
NPT_STEPS = 4000000
NVT_STEPS = 10000000
NONEQU_STEPS = 1000000

INPUT_DIR = os.path.abspath("./input_data")
OUTPUT_DIR = os.path.abspath("./transport_results")

if __name__ == '__main__':

    logger.info('build system')

    nonbonded_params = {}
    for mol in ['DMC', 'EC', 'LI', 'PF6']:
        with open(os.path.join(INPUT_DIR, f"{mol}/{mol}.json"), "r") as f:
            nonbonded_params[mol] = json.load(f)
    nonbonded_params['metadata'] = {
        "exp6": True,
        "s12": 0.1690015023721859,
        "disp_damping": 120.0,
        "thole": 0.39,
    }
    grofileparser = app.GromacsGroFile(os.path.join(INPUT_DIR, "solvent_salt.gro"))
    input_positions = grofileparser.positions
    unit_cell = grofileparser.getUnitCellDimensions()

    input_top, input_system = build_system(os.path.join(INPUT_DIR, "topol.top"), nonbonded_params, unit_cell)

    with temporary_cd(OUTPUT_DIR):
        logger.info('npt run')
        npt_positions, npt_box_vec = npt_run(
            input_top,
            input_system,
            input_positions,
            temperature=TEMPERATURE,
            npt_steps=NPT_STEPS,
        )
        rescale_positions, rescale_box_vec = rescale_box(npt_positions, npt_box_vec, work_dir='.')
        logger.info('nvt run')
        nvt_positions, nvt_box_vec = nvt_run(
            input_top,
            input_system,
            rescale_positions,
            rescale_box_vec,
            temperature=TEMPERATURE,
            work_dir=OUTPUT_DIR,
            nvt_steps=NVT_STEPS,
        )
        logger.info('nonequ run')
        nonequ_run(
            input_top,
            input_system,
            nvt_positions,
            nvt_box_vec,
            temperature=TEMPERATURE,
            work_dir=OUTPUT_DIR,
            nonequ_steps=NONEQU_STEPS,
        )

        vis = viscosity_calc(OUTPUT_DIR)
        md_volume, md_temperature = volume_calc(OUTPUT_DIR)
        logger.info('viscosity: %.3f', vis)

        species_mass_dict = {
            "PF6": [30.9740, 18.9980, 18.9980, 18.9980, 18.9980, 18.9980, 18.9980],
            "LI": [6.941],
            "EC": [12.0110, 1.0080, 1.0080, 12.0110, 1.0080, 1.0080, 15.9990, 12.0110, 15.9990, 15.9990],
            "DMC": [
                12.0110, 15.9990, 12.0110, 15.9990, 15.9990, 12.0110, 1.0080, 1.0080, 1.0080, 1.0080, 1.0080, 1.0080
            ],
        }  # anion, cation, solvent
        species_number_dict = {"PF6": 69, "LI": 69, "EC": 345, "DMC": 505}
        species_charges_dict = {"PF6": -1, "LI": 1, "EC": 0, "DMC": 0}
        nvt_positions = dcd_read('nvt.dcd')
        results = onsager_calc(species_mass_dict, species_number_dict, species_charges_dict, md_volume, vis,
                               md_temperature, nvt_positions)
        results['components'] = species_number_dict
        results['viscosity'] = vis
        with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
