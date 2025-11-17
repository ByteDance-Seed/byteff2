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
import random
from glob import glob

import pyscf
from ase import Atoms
from gpu4pyscf.dft import rks
from rdkit import Chem

from bytemol.core import Molecule
from bytemol.units import simple_unit as unit

random.seed(42)
periodic_table = Chem.GetPeriodicTable()
atomnum_elem = {i: periodic_table.GetElementSymbol(i) for i in range(1, 119)}

xc = 'WB97M-V'
basis = 'def2-TZVPD'


def calc_energy(atoms: Atoms, charge: int):
    pyscf_mol = pyscf.M()
    pyscf_mol.atom = [[atomnum_elem[an], ap] for an, ap in zip(atoms.get_atomic_numbers(), atoms.get_positions())]
    pyscf_mol.unit = 'Angstrom'
    pyscf_mol.charge = charge
    pyscf_mol.basis = basis
    pyscf_mol.build()
    mf = rks.RKS(pyscf_mol, xc=xc).density_fit(auxbasis="def2-universal-jkfit")
    mf.conv_tol = 1e-10
    mf.max_cycle = 50
    mf.kernel()
    assert mf.converged
    return unit.Hartree_to_kcal_mol(mf.e_tot)


def calc_cluster(cluster_dir: str):

    if os.path.exists(os.path.join(cluster_dir, 'energy.json')):
        return

    cluster_energy = []
    monomer_energy = []

    # load mols
    files = glob(os.path.join(cluster_dir, '*.xyz'))
    mols = [Molecule(fp) for fp in files]

    for i in range(mols[0].nconfs):
        total_charge = 0
        combined_atoms = None
        me = 0.
        for mol in mols:
            # calc monomer energy
            atoms = mol.conformers[i].to_ase_atoms()
            charge = sum(mol.formal_charges)
            energy = calc_energy(atoms, charge)
            me += energy
            total_charge += charge
            if combined_atoms is None:
                combined_atoms = atoms
            else:
                combined_atoms += atoms
        monomer_energy.append(me)

        # calc cluster energy
        ce = calc_energy(combined_atoms, total_charge)
        cluster_energy.append(ce)

    with open(os.path.join(cluster_dir, 'energy.json'), 'w') as f:
        json.dump({
            'cluster_energy': cluster_energy,
            'monomer_energy': monomer_energy,
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_samples_dir', type=str, default='./data/md_samples', help='path to the sampled clusters')
    args = parser.parse_args()

    sample_root = args.md_samples_dir
    for cluster_name in os.listdir(sample_root):
        print(f'compute {cluster_name}')
        calc_cluster(os.path.join(sample_root, cluster_name))
