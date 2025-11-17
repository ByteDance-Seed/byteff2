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
import shutil

import ase.io as aio
import MDAnalysis as mda
import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from bytemol.core import Conformer, Molecule

np.random.seed(42)

simulation_map = {
    'n1': 'n1n3n4',
    'n2': 'n2n5',
    'n3': 'n1n3n4',
    'n4': 'n1n3n4',
    'n5': 'n2n5',
    'i1': 'i1i4',
    'i2': 'i2i5',
    'i3': 'i3',
    'i4': 'i1i4',
    'i5': 'i2i5',
}

all_comps = {
    'n1': {
        'EtCl': 3
    },
    'n2': {
        'BZ': 3
    },
    'n3': {
        'EtCl': 1,
        'EtSH': 1,
        'NOM': 1
    },
    'n4': {
        'An': 2,
        'NOM': 1
    },
    'n5': {
        'ACE': 1,
        'ACN': 1,
        'BZ': 1,
        'HAC': 1
    },
    'i1': {
        'EC': 3,
        'PF6': 1
    },
    'i2': {
        'DMC': 4,
        'LI': 1
    },
    'i3': {
        'EC': 2,
        'EMC': 2,
        'LI': 1
    },
    'i4': {
        'EC': 3,
        'LI': 1,
        'PF6': 1
    },
    'i5': {
        'DMC': 2,
        'FSI': 2,
        'LI': 1
    }
}


def load_traj(gro_path, dcd_path):
    u = mda.Universe(gro_path, dcd_path)
    atom_rec = {}
    res_rec = {}
    for idx, atom in enumerate(u.atoms):
        atom_rec[idx] = (atom.resname, atom.resid)
        if atom.resid not in res_rec:
            res_rec[atom.resid] = [atom.resname, []]
        assert atom.resname == res_rec[atom.resid][0]
        res_rec[atom.resid][1].append(idx)
    return u, atom_rec, res_rec


def sample_cluster(positions, box, cutoff, atom_rec, res_rec, cluster_comps, smiles_map) -> list[Molecule]:
    position = np.array(positions) % box
    tree = cKDTree(position, boxsize=box)
    if 'LI' in cluster_comps:
        while True:
            iatom = np.random.randint(0, position.shape[0])
            if atom_rec[iatom][0] == 'LI':
                break
    else:
        iatom = np.random.randint(0, position.shape[0])
    cp = position[iatom]
    atoms = tree.query_ball_point(cp, cutoff)
    ires = set([atom_rec[ia][1] for ia in atoms])
    res_count = {}
    for iresid in ires:
        name, ids = res_rec[iresid]
        if name in res_count:
            res_count[name] += 1
        else:
            res_count[name] = 1

    flag = True
    for name, num in cluster_comps.items():
        if name not in res_count or res_count[name] < num:
            flag = False
            break
    if not flag:
        return None

    mols = []
    cp = None
    cluster_rec = {k: 0 for k in cluster_comps.keys()}
    for iresid in ires:
        name, ids = res_rec[iresid]
        if name not in cluster_rec or cluster_rec[name] == cluster_comps[name]:
            continue
        if cp is None:
            cp = position[ids[0]]
        mol = Molecule.from_smiles(smiles_map[name], name=name)
        coords = position[ids] - cp
        coords = (coords + 0.5 * box) % box - 0.5 * box
        conf = Conformer(coords, mol.atomic_symbols)
        mol.append_conformers(conf)
        mols.append(mol)
        cluster_rec[name] += 1

    for k, v in cluster_rec.items():
        assert v == cluster_comps[k]
    return mols


def main(args):
    cutoff = 4.
    cluster_count = 40

    shutil.rmtree(args.md_samples_dir, ignore_errors=True)
    os.makedirs(args.md_samples_dir)

    for cluster_name in all_comps.keys():
        simulation_name = simulation_map[cluster_name]
        save_dir = os.path.join(args.md_samples_dir, f'cluster-{cluster_name}')
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(args.config_root, f'{simulation_name}.json')) as f:
            smiles_map = json.load(f)['smiles']

        natom_map = {}
        for k, v in smiles_map.items():
            natom_map[k] = Molecule.from_smiles(v).natoms

        print('sample', cluster_name)

        cluster_comps = all_comps[cluster_name]
        gro_path = os.path.join(args.trj_root, 'results_' + simulation_name, 'solvent_salt.gro')
        dcd_path = os.path.join(args.trj_root, 'results_' + simulation_name, 'npt.dcd')
        cluster_names = []
        for k, v in cluster_comps.items():
            cluster_names.extend([k] * v)

        # load trajectory
        u, atom_rec, res_rec = load_traj(gro_path, dcd_path)
        all_clusters: list[list[Molecule]] = []
        for ts in u.trajectory[2000::10]:
            for _ in range(100):
                mols = sample_cluster(ts.positions, ts.dimensions[:3], cutoff, atom_rec, res_rec, cluster_comps,
                                      smiles_map)
                if mols is None:
                    continue
                else:
                    ordered_mols = []
                    for name in cluster_names:
                        for imol, mol in enumerate(mols):
                            if mol.name == name:
                                ordered_mols.append(mol)
                                break
                        mols.pop(imol)
                    assert len(ordered_mols) == len(cluster_names)
                    all_clusters.append(ordered_mols)
                    break

            if len(all_clusters) > cluster_count:
                break

        if len(all_clusters) < cluster_count:
            raise RuntimeError(f'Only {len(all_clusters)} clusters found')

        conformers = [[] for _ in cluster_names]
        for i in range(cluster_count):
            for j in range(len(cluster_names)):
                conformers[j].append(all_clusters[i][j].conformers[0])
        # write results
        for i, name in enumerate(cluster_names):
            mol = all_clusters[0][i]
            mol._conformers = conformers[i]
            mol.to_xyz(os.path.join(save_dir, f'{mol.name}_{i}.xyz'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_root', type=str, default='./data/md_configs')
    parser.add_argument('--trj_root', type=str, default='./data/md_trj')
    parser.add_argument('--md_samples_dir',
                        type=str,
                        default='./data/md_samples',
                        help='where to save sampled clusters')
    args = parser.parse_args()

    main(args)
