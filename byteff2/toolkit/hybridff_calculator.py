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

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import CalculationFailed

from byteff2.data import ClusterData
from byteff2.model import HybridFF
from bytemol.core import Molecule
from bytemol.toolkit.asetool import OptimizerConfig, OptimizerNotConvergedException, optimize
from bytemol.toolkit.asetool.basecalculator import BaseCalculator

logger = logging.getLogger(__name__)


class HybridFFCalculator(BaseCalculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, mols: list[Molecule], forcefield: HybridFF, edge3d_rcut=0.0, cluster=True):
        super().__init__()

        self.mols = mols
        self.data = ClusterData('test', [mol.get_mapped_smiles() for mol in mols],
                                confdata={'coords': np.random.rand(1, sum([mol.natoms for mol in mols]), 3)},
                                max_n_confs=1)
        self.edge3d_rcut = edge3d_rcut
        self.forcefield = forcefield
        self.cluster = cluster

    def _calculate_without_restraint(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Calculate force field energy and force in PySCFCalculatorImp.
        """

        coords = torch.tensor(coords, device=self.data.coords.device, dtype=self.data.coords.dtype).unsqueeze(1)
        self.data.coords = coords
        if self.edge3d_rcut > 0.:
            self.data.set_edge_3d(self.edge3d_rcut)

        preds = self.forcefield(self.data, cluster=self.cluster)
        # s = ''
        # for k, v in preds['ff_parameters'].items():
        #     if 'energy_cluster' in k and not k.split('.')[1] == 'energy_cluster':
        #         s += f"{k.split('.')[1]} {torch.tensor(v).sum().item():.2f} "
        # print(s)

        sufix = '_cluster' if self.cluster else ''
        if 'MultipoleInt.D_ind' + sufix in preds['ff_parameters'] and preds['ff_parameters']['MultipoleInt.D_ind' +
                                                                                             sufix] is not None:
            self.data['MultipoleInt.D_ind' + sufix] = preds['ff_parameters']['MultipoleInt.D_ind' +
                                                                             sufix].clone().detach()

        if self.cluster:
            return preds['energy_cluster'][0, 0].detach().numpy(), preds['forces_cluster'].squeeze(1).detach().numpy()
        else:
            return preds['energy'][0, 0].detach().numpy(), preds['forces'].squeeze(1).detach().numpy()


def hybridff_ase_opt(mols: list[Molecule],
                     model: HybridFF,
                     conformer=0,
                     position_restraint=0.,
                     save_trj=False,
                     cluster=True,
                     constraints=None,
                     max_iterations=1000) -> tuple[list[Molecule], bool]:

    symbols, positions = [], []
    charge = 0
    natoms = 0
    for mol in mols:
        symbols += mol.atomic_symbols
        positions.append(mol.conformers[conformer].coords)
        charge += sum(mol.formal_charges)
        natoms += mol.natoms
    positions = np.concatenate(positions, axis=0)
    atoms = Atoms(symbols=symbols, positions=positions)

    calc = HybridFFCalculator(mols, model, cluster=cluster)

    if position_restraint > 0.:
        calc.set_position_restraints(list(range(natoms)), force_constant=position_restraint, target=positions)

    if save_trj:
        calc.init_trajectory('optim_ff.xyz' if cluster else 'optim_ff_sep.xyz')

    optimizer_config = OptimizerConfig({
        "common": {
            "fmax": 0.01,
            "max_iterations": max_iterations,
            "logfile": 'relax.log',
        },
        "optimizer": {
            "type": "bfgs",
            "params": {
                "maxstep": 0.1,
                "alpha": 70,
            }
        }
    })

    try:
        relaxed = optimize(atoms, config=optimizer_config, calculator=calc, verbose=True, constraints=constraints)
    except (OptimizerNotConvergedException, CalculationFailed) as e:
        logger.info(f'optimization failed, {e}')
        return None

    positions = relaxed.positions
    begin_id = 0
    new_mols = []
    for mol in mols:
        new_mol = mol.copy(keep_conformers=False)
        new_mol._conformers = [mol.conformers[0].copy()]
        new_mol.conformers[-1].coords = positions[begin_id:begin_id + mol.natoms]
        begin_id += mol.natoms
        new_mols.append(new_mol)

    return new_mols
