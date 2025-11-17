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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set rcParams for academic paper
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],
    'font.size': 12,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (5, 4),
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'savefig.dpi': 150,
}

plt.rcParams.update(params)

data = pd.read_csv('data_fig3d.csv')
dd = np.abs(data['Exp Density [g/mL]'] - data['Sim Density [g/mL]'])

plt.scatter(1 - data['Similarity'], dd, s=1)
plt.xlabel('1 - Similarity')
plt.ylabel('Absolute Error [g/mL]')
plt.tight_layout()
plt.savefig('sim_error_scatter.png')
plt.close()

bins = np.linspace(0, 0.15, 10)
sum_vals, count_vals = np.zeros(len(bins) + 1), np.zeros(len(bins) + 1)
for d, s in zip(dd, data['Similarity']):
    ibin = np.digitize(1 - s, bins)
    sum_vals[ibin] += d
    count_vals[ibin] += 1

w = bins[1:] - bins[:-1]
x = bins[:-1] + w / 2
plt.bar(x, sum_vals[1:-1] / count_vals[1:-1], w, alpha=0.7, edgecolor='black')
plt.xlabel('1 - Similarity')
plt.ylabel('Density MAE [g/mL]')
plt.tight_layout()
plt.savefig('sim_error_stat.png')
