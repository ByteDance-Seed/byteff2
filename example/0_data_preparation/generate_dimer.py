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
import os

import numpy as np
from scipy.spatial.transform import Rotation

from bytemol.core import Molecule
from bytemol.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser('create dimer for electrolyte')
parser.add_argument(
    '--mol1',
    type=str,
    default='./ACT.xyz',
    help='mol1 xyz',
)
parser.add_argument(
    '--mol2',
    type=str,
    default='./EC.xyz',
    help='mol2 xyz',
)
parser.add_argument(
    '--save_dir',
    type=str,
    default='./ACT_EC_dimer',
    help='directory for saving xyz',
)
parser.add_argument('--nconfs', type=int, default=100, help='number of conformers')
parser.add_argument('--min_dist', type=float, default=1.5, help='minimum distance between two atoms in Angstrom')
parser.add_argument('--max_dist', type=float, default=10., help='maximum displacement in Angstrom')
parser.add_argument('--noise_std', type=float, default=0.05, help='noise adding to coords in Angstrom')
args = parser.parse_args()

np.random.seed(42)


def calc_min_dist2(coords1: np.ndarray, coords2: np.ndarray):
    c11 = coords1[:, np.newaxis]
    c11 = np.tile(c11, (1, coords2.shape[0], 1))
    c21 = coords2[np.newaxis, ...]
    c21 = np.tile(c21, (coords1.shape[0], 1, 1))

    dist = c21 - c11
    dist2 = np.square(dist).sum(-1).min()
    return dist2


def perturb_coords(coords1: np.ndarray, coords2: np.ndarray, nconfs=100, min_dist=1.5, max_dist=10.):

    coords_new = []
    disps = []
    while True:
        d = np.random.uniform(0, max_dist)
        disp = np.random.uniform(-1, 1, (1, 3))
        disp = disp / np.sqrt(np.sum(np.square(disp))) * d
        rot = Rotation.random()
        c2_new = rot.apply(coords2) + disp + np.random.normal(0, args.noise_std, coords2.shape)
        c1_new = coords1 + np.random.normal(0, args.noise_std, coords1.shape)
        if calc_min_dist2(c1_new, c2_new) < min_dist**2:
            continue

        disps.append(d)
        coords_new.append([c1_new.copy(), c2_new.copy()])
        if len(coords_new) == nconfs:
            break

    ids = np.argsort(disps)
    coords_sorted = []
    for i in ids:
        coords_sorted.append(coords_new[i])
    return coords_sorted


if __name__ == '__main__':
    mol1 = Molecule.from_xyz(args.mol1)
    mol2 = Molecule.from_xyz(args.mol2)
    os.makedirs(args.save_dir, exist_ok=True)
    coords = perturb_coords(
        mol1.conformers[0].coords,
        mol2.conformers[0].coords,
        args.nconfs,
        args.min_dist,
        args.max_dist,
    )
    for ci, (c1, c2) in enumerate(coords):
        conf_dir = os.path.join(args.save_dir, f'conf_{ci}')
        os.makedirs(conf_dir, exist_ok=True)
        mol1.conformers[0].coords = c1
        mol1.to_xyz(os.path.join(conf_dir, f'{mol1.name}.xyz'), append=True)
        mol2.conformers[0].coords = c2
        mol2.to_xyz(os.path.join(conf_dir, f'{mol2.name}.xyz'), append=True)
