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
from glob import glob

import numpy as np
import pandas as pd

from byteff2.data import ClusterData, collate_data
from byteff2.toolkit.hybridff_calculator import hybridff_ase_opt
from byteff2.train import load_model
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path, temporary_cd


def get_ff_energy(model, _mols, conformer=0):

    coords_raw = [mol.conformers[conformer].coords.reshape(1, -1, 3) for mol in _mols]
    coords = np.concatenate(coords_raw, axis=1)

    x = np.mean(coords_raw[1], axis=1) - np.mean(coords_raw[0], axis=1)
    x = np.linalg.norm(x, axis=-1)

    _data = ClusterData('test', [mol.get_mapped_smiles() for mol in _mols], confdata={'coords': coords}, max_n_confs=1)
    _data = collate_data([_data])

    preds = model(_data, cluster=True)

    ff_results = {}
    ff_results['SEP'] = preds['energy'].sum(dim=0)
    ff_results['TOTAL'] = preds['energy_cluster'].sum(dim=0) - preds['energy'].sum(dim=0)
    ff_results['POLARIZATION'] = preds['ff_parameters']['POLARIZATION'].sum(dim=0)
    ff_results['ELEC'] = preds['ff_parameters']['ELEC'].sum(dim=0)
    ff_results['CHARGE_TRANSFER'] = preds['ff_parameters']['CHARGE_TRANSFER'].sum(dim=0)

    ff_results['PAULI'] = preds['ff_parameters']['PAULI'].sum(dim=0)
    ff_results['DISP'] = preds['ff_parameters']['DISP'].sum(dim=0)

    for k in ff_results:
        ff_results[k] = ff_results[k].detach().tolist()
    ff_results['ELEC_PAULI'] = np.array(ff_results['ELEC']) + np.array(ff_results['PAULI'])
    ff_results['ELEC_PAULI_DISP'] = np.array(ff_results['ELEC']) + np.array(ff_results['PAULI']) + np.array(
        ff_results['DISP'])
    return ff_results


if __name__ == '__main__':

    position_restraint = 1.0
    model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
    trained_model = load_model(os.path.dirname(model_dir))

    src = os.path.abspath('./cluster_data')
    save_dir = os.path.abspath('./cluster_results')

    os.makedirs(save_dir, exist_ok=True)

    names = []
    props = ['ELEC_PAULI', 'DISP', 'POLARIZATION', 'CHARGE_TRANSFER', 'TOTAL', 'CONF', 'TOTAL + CONF']

    results = {k: [] for k in props}
    results['RMSD'] = []

    for dirname in sorted(os.listdir(src)):

        print(dirname)
        save_dd = os.path.join(save_dir, dirname)
        os.makedirs(save_dd, exist_ok=True)

        json_fp = f'{src}/{dirname}/EDA.json'
        with open(json_fp) as file:
            qm_results = json.load(file)
            qm_results['ELEC_PAULI'] = np.array(qm_results['ELEC_PAULI'])
            qm_results['CHARGE_TRANSFER'] = np.array(qm_results['CHARGE TRANSFER'])
            qm_results['CONF'] = sum(qm_results['CONF'])
            qm_results['TOTAL + CONF'] = qm_results['CONF'] + qm_results['TOTAL']

        names.append(dirname)
        files = glob(os.path.join(src, dirname, '*.xyz'))
        mols = [Molecule(fp) for fp in files if not 'optim' in fp]

        # evaluate energy on QM conformations
        ff_results0 = get_ff_energy(trained_model, mols, conformer=-1)

        # FF relaxation for cluster
        with temporary_cd(save_dd):
            new_mols = hybridff_ase_opt(mols,
                                        model=trained_model,
                                        conformer=-1,
                                        save_trj=True,
                                        position_restraint=position_restraint)
            rmsd = 0.
            for i, mol in enumerate(new_mols):
                mol.to_xyz(f'{mol.name}_{i}.xyz')
                rmsd += np.sqrt(np.sum((mols[i].conformers[-1].coords - mol.conformers[-1].coords)**2))

        # evaluate energy on FF relaxed conformations for cluster
        ff_results1 = get_ff_energy(trained_model, new_mols, conformer=-1)

        # FF relaxation for separte molecules
        with temporary_cd(save_dd):
            new_mols = hybridff_ase_opt(new_mols,
                                        model=trained_model,
                                        conformer=-1,
                                        save_trj=True,
                                        position_restraint=position_restraint,
                                        cluster=False)
            for i, mol in enumerate(new_mols):
                mol.to_xyz(f'{mol.name}_{i}_sep.xyz')

        # evaluate energy on FF relaxed conformations for separte molecules
        ff_results2 = get_ff_energy(trained_model, new_mols, conformer=-1)

        ff_results1['CONF'] = sum(ff_results1['SEP']) - sum(ff_results2['SEP'])
        ff_results1['TOTAL + CONF'] = ff_results1['CONF'] + ff_results1['TOTAL'][0]

        for prop in props:
            if 'CONF' in prop:
                if 'CONF' in qm_results:
                    results[prop].append(f'({qm_results[prop]:.2f}) {ff_results1[prop]:.2f}')
                else:
                    results[prop].append(f'{ff_results1[prop]:.2f}')
            else:
                results[prop].append(f'({qm_results[prop]:.2f}) {ff_results0[prop][0]:.2f} {ff_results1[prop][0]:.2f}')
        results['RMSD'].append(f'{rmsd:.2f}')

    data = {'names': names}
    data.update(results)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
