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

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from byteff2.data import IMDataset, collate_data
from byteff2.train import load_model
from bytemol.utils import get_data_file_path


def get_interaction_energy(preds_energy, label_data):
    pred_cluster = preds_energy['energy_cluster']
    nmols = label_data.get_count('mol', idx=None, cluster=True)
    batches = torch.arange(nmols.shape[0], device=nmols.device).repeat_interleave(nmols).unsqueeze(-1).expand(
        -1, pred_cluster.shape[1])
    pred_single = torch.zeros_like(pred_cluster).scatter_add_(0, batches, preds_energy['energy'])

    le = label_data.total_int_energy
    return pred_cluster - pred_single, le


if __name__ == '__main__':

    device = 'cpu'  # 'cpu' or 'cuda'
    model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
    model = load_model(os.path.dirname(model_dir)).to(device)

    dataset_path = get_data_file_path('valid_data/dataset_config.yaml', 'byteff2')
    dataset = IMDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_data, drop_last=False)

    all_label, all_pred = [], []

    for data in dataloader:
        data = data.to(device)
        preds = model(data, cluster=True)
        pe, label = get_interaction_energy(preds, data)
        pe = pe[data.confmask_cluster > 0.5].detach().flatten().tolist()
        label = label[data.confmask_cluster > 0.5].detach().flatten().tolist()
        all_label.extend(label)
        all_pred.extend(pe)
        del preds, data

    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(all_label, all_pred, s=1)
    plt.axis('square')
    plt.plot([-50, 150], [-50, 150], 'k--', zorder=0)
    plt.xlim(-50, 150)
    plt.ylim(-50, 150)
    plt.xlabel(r'DFT $U_\mathrm{int}$ [kcal/mol]')
    plt.ylabel(r'ByteFF-Pol $U_\mathrm{int}$ [kcal/mol]')
    plt.tight_layout()
    plt.savefig('Fig2a.png', dpi=150)
