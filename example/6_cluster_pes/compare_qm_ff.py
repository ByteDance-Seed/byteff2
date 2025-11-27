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

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef

from byteff2.data import ClusterData, collate_data
from byteff2.train import load_model
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path

# Set rcParams for academic paper
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (4, 4),
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'savefig.dpi': 150,
}

plt.rcParams.update(params)


def main(args):

    pearson = PearsonCorrCoef()
    mae = MeanAbsoluteError()

    model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
    trained_model = load_model(os.path.dirname(model_dir))

    root = args.md_samples_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for cluster_name in os.listdir(root):
        src = os.path.join(root, cluster_name)
        if not os.path.exists(os.path.join(src, 'energy.json')):
            continue
        print(cluster_name)
        with open(os.path.join(src, 'energy.json')) as file:
            data = json.load(file)
            qm_energies = np.array(data['cluster_energy']) - np.array(data['monomer_energy'])

        files = glob(os.path.join(src, '*.xyz'))
        mols = [Molecule(fp) for fp in files]

        coords_raw = [np.array(mol.get_confdata('coords')) for mol in mols]
        coords = np.concatenate(coords_raw, axis=1)
        data = ClusterData('test', [mol.get_mapped_smiles() for mol in mols],
                           confdata={'coords': coords},
                           max_n_confs=mols[0].nconfs)
        data = collate_data([data])

        preds = trained_model(data, cluster=True)
        ff_e = preds['energy_cluster'].sum(dim=0) - preds['energy'].sum(dim=0)
        ff_energies = ff_e.detach().numpy()

        plt.scatter(qm_energies, ff_energies)
        plt.xlabel(r'DFT $U_\mathrm{int}$ (kcal/mol)')
        plt.ylabel(r'ByteFF-Pol $U_\mathrm{int}$ (kcal/mol)')
        l1, r1 = plt.xlim()
        l2, r2 = plt.ylim()
        l = min(l1, l2)
        r = max(r1, r2)
        plt.plot([l, r], [l, r], 'k--', zorder=1)
        plt.axis('square')
        plt.title(f'{cluster_name}')
        pearson_v = pearson(torch.tensor(ff_energies), torch.tensor(qm_energies))
        mae_v = mae(torch.tensor(ff_energies), torch.tensor(qm_energies))

        plt.gca().text(l + 0.45 * (r - l),
                       l + 0.03 * (r - l),
                       f'MAE={mae_v.item():.2f}\nPearson={pearson_v.item():.2f}',
                       fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/energy_compare_{cluster_name}.png')
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--md_samples_dir',
                        type=str,
                        default='./data/md_sample_examples',
                        help='path to the sampled clusters')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./cluster_pes',
                        help='path to the sampled clusters')
    args = parser.parse_args()

    main(args)
