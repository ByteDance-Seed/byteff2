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

import matplotlib.pyplot as plt
import numpy as np

from byteff2.data import ClusterData, collate_data
from byteff2.train import load_model
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path

if __name__ == '__main__':

    model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
    model = load_model(os.path.dirname(model_dir))

    src = os.path.abspath('./dimer_data')
    save_dir = os.path.abspath('dimer_results')

    os.makedirs(save_dir, exist_ok=True)
    for dirname in os.listdir(src):

        print(dirname)

        with open(f'{src}/{dirname}/EDA.json') as file:
            qm_results = json.load(file)

        names = dirname.split('_')[:2]

        mols = [Molecule(os.path.join(src, dirname, f'{name}_{i}.xyz')) for i, name in enumerate(names)]
        coords_raw = [np.array(mol.get_confdata('coords')) for mol in mols]
        coords = np.concatenate(coords_raw, axis=1)

        x = np.mean(coords_raw[1], axis=1) - np.mean(coords_raw[0], axis=1)
        x = np.linalg.norm(x, axis=-1)

        data = ClusterData('test', [mol.get_mapped_smiles() for mol in mols],
                           confdata={'coords': coords},
                           max_n_confs=mols[0].nconfs)
        data = collate_data([data])

        preds = model(data, cluster=True)

        ff_results = {}
        ff_results['TOTAL'] = preds['energy_cluster'].sum(dim=0) - preds['energy'].sum(dim=0)
        ff_results['POLARIZATION'] = preds['ff_parameters']['POLARIZATION'].sum(dim=0)
        ff_results['ELEC'] = preds['ff_parameters']['ELEC'].sum(dim=0)
        ff_results['PAULI'] = preds['ff_parameters']['PAULI'].sum(dim=0)
        ff_results['DISP'] = preds['ff_parameters']['DISP'].sum(dim=0)
        ff_results['CHARGE_TRANSFER'] = preds['ff_parameters']['CHARGE_TRANSFER'].sum(dim=0)

        for k in ff_results:
            ff_results[k] = ff_results[k].detach().tolist()

        ff_results['ELEC_PAULI'] = np.array(ff_results['ELEC']) + np.array(ff_results['PAULI'])
        ff_results['ELEC_PAULI_DISP'] = np.array(ff_results['ELEC']) + np.array(ff_results['PAULI']) + np.array(
            ff_results['DISP'])
        qm_results['ELEC_PAULI'] = np.array(qm_results['ELEC_PAULI'])
        qm_results['TOTAL'] = np.array(qm_results['TOTAL'])
        qm_results['CHARGE_TRANSFER'] = qm_results.pop('CHARGE TRANSFER')

        save_results = {}

        for k, v in ff_results.items():
            save_results[k] = list(v)
            if k in qm_results:
                save_results[k + '_QM'] = list(qm_results[k])

        save_results['distance'] = x.tolist()

        with open(f"{save_dir}/{dirname}.json", 'w') as file:
            json.dump(save_results, file, indent=2)

        # props = ['TOTAL', 'ELEC_PAULI', 'POLARIZATION', 'DISP', 'ELEC']
        # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']  # , 'tab:red'

        props = ['TOTAL', 'ELEC_PAULI', 'POLARIZATION', 'DISP', "CHARGE_TRANSFER"]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']  # , 'tab:red'

        for prop, c in zip(props, colors):
            plt.plot(x, ff_results[prop], c=c, label=f'{prop}')
            plt.plot(x, qm_results[prop], c=c, linestyle='--')

        plt.xlabel('distance [A]')
        plt.ylabel('interaction energy [kcal/mol]')
        plt.title(dirname.replace('_', '-'))
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{dirname}.jpg'))
        plt.close()
