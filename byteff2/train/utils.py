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

import logging
import os

import numpy as np
import torch
import yaml

from byteff2.data import GraphData
from byteff2.model import HybridFF
from byteff2.toolkit import ffparams_to_tfs
from bytemol.core import Molecule

logger = logging.getLogger(__name__)


def load_model(model_config, ckpt: str = None) -> HybridFF:

    if isinstance(model_config, str):
        config_path = os.path.join(model_config, "fftrainer_config_in_use.yaml")
        with open(config_path) as file:
            config = yaml.safe_load(file)
        config = config['model']
        config.pop('check_point', None)
        model = HybridFF(**config)

        sd = torch.load(
            os.path.join(model_config, 'optimal.pt'),
            weights_only=True,
            map_location='cpu',
        )
        model.load_state_dict(sd['model_state_dict'])

    else:
        assert isinstance(model_config, dict)
        model = HybridFF(**model_config)

    if ckpt is not None:
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd['model_state_dict'])
        # model.load_state_dict(sd['model_state_dict'], strict=False)

    model.eval()
    return model


def get_nb_params(model: HybridFF, mol: Molecule, relax=True):

    metadata = {
        'exp6': True,
        's12': model.ff_block.ff_layers['Exp6Pol'].s12 / 10. * 4.184**(1 / 12),  # in kcal^(1/12) * nm
        'disp_damping': model.ff_block.ff_layers['Exp6Pol'].disp_damping_factor,
        'thole': model.ff_block.ff_layers['Exp6Pol'].dipole_solver.a,
    }

    data = GraphData('test', mol.get_mapped_smiles(), record_nonbonded_all=False)
    preds = model(data, skip_ff=True)
    ffparams = preds['ff_parameters']

    c6 = ffparams['PreExp6Pol.c6']
    r0 = ffparams['PreExp6Pol.rvdw']
    lamb = ffparams['PreExp6Pol.lambda']
    eps = ffparams['PreExp6Pol.eps']
    lamb = lamb.reshape(-1).detach().tolist()
    eps = (eps * 4.184).reshape(-1).detach().tolist()

    params = {
        'charge': ffparams['PreChargeVolume.charges'].flatten().detach().tolist(),
        'alpha': (ffparams['PreExp6Pol.alpha'] * 1e-3).flatten().detach().tolist(),  # nm^3
        'pol_damping': (ffparams['PreExp6Pol.pol_damping'] * 1e-3).flatten().detach().tolist(),  # nm^3
        'lamb': lamb,
        'eps': eps,  # kJ/mol
        'Rvdw': (r0 * 0.1).flatten().detach().tolist(),  # nm
        'C6': (c6 * 4.184 * 1e-6).flatten().detach().tolist(),  # kJ/mol * nm^6
        'ct_eps': (ffparams['PreExp6Pol.ct_eps'] * 4.184 * 1e-4).flatten().detach().tolist(),  # kj/mol * nm^4
        'ct_lamb': ffparams['PreExp6Pol.ct_lamb'].flatten().detach().tolist(),
    }

    # add fake nonbonded params to itp
    ffparams['PreLJEs.sigma'] = torch.zeros(len(params['charge']))
    ffparams['PreLJEs.epsilon'] = torch.zeros(len(params['charge']))
    ffparams['PreLJEs.charge'] = ffparams['PreChargeVolume.charges'].flatten().detach()
    tfs = ffparams_to_tfs(ffparams, data, mol, mol_name=mol.name)

    if mol.name == 'PF6':
        mol.conformers[0].coords = np.array([
            [-0.082, -0.291, 0.000],
            [-0.082, -0.291, 1.630],
            [-1.712, -0.291, 0.000],
            [-0.082, 1.339, 0.000],
            [-0.082, -0.291, -1.630],
            [1.548, -0.291, 0.000],
            [-0.082, -1.921, 0.000],
        ])
    return metadata, params, tfs, mol
