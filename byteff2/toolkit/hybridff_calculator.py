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

from ase import Atoms
from ase.calculators.calculator import CalculationFailed
import numpy as np
import torch

from byteff2.bytemol.core import Molecule
from byteff2.bytemol.core.rkutil.conformer import find_hypervalent_centers
from byteff2.bytemol.toolkit.asetool import optimize, OptimizerConfig, OptimizerNotConvergedException
from byteff2.bytemol.toolkit.asetool.basecalculator import BaseCalculator
from byteff2.data import ClusterData
from byteff2.model import HybridFF
from byteff2.toolkit.xtb_calculator import patch_hypervalent_bonded_params_for_write, xtb_optimized_coords


logger = logging.getLogger(__name__)


class HybridFFCalculator(BaseCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        mols: list[Molecule],
        forcefield: HybridFF,
        edge3d_rcut=0.0,
        cluster=True,
        conformer=0,
        sep=False,
    ):
        super().__init__()

        self.mols = mols
        self.natoms = sum(mol.natoms for mol in self.mols)
        self.edge3d_rcut = edge3d_rcut
        self.forcefield = forcefield.eval()
        self.cluster = cluster
        self.data = None
        self.mol_energy = None
        self.sep = sep
        self.separate_energy = {}
        self.separate_forces = {}
        self.device = next(self.forcefield.parameters()).device

        self._cached_node_h = None
        self._cached_edge_h = None
        self._cached_ffparams = None

        self._init_forcefield_cache(mols, conformer)

    def _initial_coords(self, mols: list[Molecule], conformer=0, mm_flag=True) -> np.ndarray:
        """Initial cluster coords for forcefield-cache init.

        Hypervalent molecules need xtb-gfn2 optimized geometry so the
        write-time bond/angle patch in
        ``patch_hypervalent_bonded_params_for_write`` can read sensible
        bond lengths / angles. Other molecules use their existing conformer
        if available, else a random placeholder.
        """
        coords = []
        for mol in mols:
            if find_hypervalent_centers(mol.rkmol) and mm_flag:
                coords.append(xtb_optimized_coords(mol, conformer))
            elif mol.conformers:
                coords.append(mol.conformers[conformer].coords)
            else:
                coords.append(np.random.rand(mol.natoms, 3))
        return np.concatenate(coords, axis=0)[None, ...]

    def _has_mm_bonded_layer(self) -> bool:
        ff_block = getattr(self.forcefield, "ff_block", None)
        ff_layers = getattr(ff_block, "ff_layers", {})
        return "MMBonded" in ff_layers or "MMBondedConj" in ff_layers

    def _init_forcefield_cache(self, mols: list[Molecule], conformer: int):
        mm_flag = self._has_mm_bonded_layer()

        self.data = ClusterData(
            "test",
            [mol.get_mapped_smiles() for mol in mols],
            confdata={"coords": self._initial_coords(mols, conformer, mm_flag)},
            max_n_confs=1,
        )
        self.data = self.data.to(self.device)
        with torch.no_grad():
            node_h, edge_h, xs = self.forcefield.graph_block(self.data)
            ffparams = {"Graph2D.xs": xs}
            ffparams = self.forcefield.preff_block(self.data, node_h, edge_h, ffparams, do_patch=False)
            if mm_flag:
                patch_hypervalent_bonded_params_for_write(ffparams, self.data)

        self._cached_node_h = node_h
        self._cached_edge_h = edge_h
        self._cached_ffparams = ffparams

    def _calculate_with_cached_params(self) -> tuple[np.ndarray, np.ndarray]:
        ffparams = dict(self._cached_ffparams)
        with torch.no_grad():
            energy, forces = self.forcefield.ff_block(
                self.data, self._cached_node_h, self._cached_edge_h, ffparams, cluster=False
            )
            self.mol_energy = energy[:, 0].detach().cpu().numpy()
            if self.cluster:
                energy, forces = self.forcefield.ff_block(
                    self.data, self._cached_node_h, self._cached_edge_h, ffparams, cluster=True
                )

        if self.sep and self.cluster:
            self._update_separate_terms(ffparams)

        d_ind_key = "MultipoleInt.D_ind" + ("_cluster" if self.cluster else "")
        if d_ind_key in ffparams and ffparams[d_ind_key] is not None:
            self.data[d_ind_key] = ffparams[d_ind_key].clone().detach()
        return energy[:, 0].sum().detach().cpu().numpy(), forces.squeeze(1).detach().cpu().numpy()

    def _update_separate_terms(self, ffparams: dict[str, torch.Tensor]):
        self.separate_energy.clear()
        self.separate_energy["INTER_VdW"] = 0.0

        disp = ffparams.get("DISP")
        pauli = ffparams.get("PAULI")
        elec = ffparams.get("ELEC")
        pol = ffparams.get("POLARIZATION")
        if disp is not None and pauli is not None and elec is not None and pol is not None:
            self.separate_energy["DISP"] = disp.detach().cpu().numpy().sum()
            self.separate_energy["PAULI"] = pauli.detach().cpu().numpy().sum()
            self.separate_energy["ELEC"] = elec.detach().cpu().numpy().sum()
            self.separate_energy["POLARIZATION"] = pol.detach().cpu().numpy().sum()
            self.separate_energy["INTER_CHARGE"] = self.separate_energy["ELEC"] + self.separate_energy["POLARIZATION"]
            self.separate_energy["INTER_VdW"] = self.separate_energy["PAULI"] + self.separate_energy["DISP"]

        charge_transfer = ffparams.get("CHARGE_TRANSFER")
        self.separate_energy["CHARGE_TRANSFER"] = (
            0.0 if charge_transfer is None else charge_transfer.detach().cpu().numpy().sum()
        )
        self.separate_energy["INTER_VdW"] += self.separate_energy["CHARGE_TRANSFER"]

        nnp_cluster = ffparams.get("NNPLayer.energy_cluster")
        nnp_mono = ffparams.get("NNPLayer.energy")
        if nnp_cluster is not None and nnp_mono is not None:
            self.separate_energy["INTER_NNP"] = (
                nnp_cluster.detach().cpu().numpy().sum() - nnp_mono.detach().cpu().numpy().sum()
            )
            self.separate_energy["INTER_VdW"] += self.separate_energy["INTER_NNP"]

    def _calculate_without_restraint(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate force field energy and force in PySCFCalculatorImp."""

        coords = torch.tensor(coords, device=self.data.coords.device, dtype=self.data.coords.dtype).unsqueeze(1)
        self.data.coords = coords
        if self.edge3d_rcut > 0.0:
            self.data.set_edge_3d(self.edge3d_rcut)

        return self._calculate_with_cached_params()

    def get_mols_energies(self) -> np.ndarray:
        return self.mol_energy

    def get_separate_terms(self) -> dict:
        return self.separate_energy, self.separate_forces


def hybridff_ase_opt(
    mols: list[Molecule],
    model: HybridFF,
    conformer=0,
    position_restraint=0.0,
    save_trj=False,
    cluster=True,
    constraints=None,
    max_iterations=1000,
) -> tuple[list[Molecule], bool]:

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

    calc = HybridFFCalculator(mols, model, cluster=cluster, conformer=conformer)

    if position_restraint > 0.0:
        calc.set_position_restraints(list(range(natoms)), force_constant=position_restraint, target=positions)

    if save_trj:
        calc.init_trajectory("optim_ff.xyz" if cluster else "optim_ff_sep.xyz")

    optimizer_config = OptimizerConfig(
        {
            "common": {
                "fmax": 0.01,
                "max_iterations": max_iterations,
                "logfile": "relax.log",
            },
            "optimizer": {
                "type": "bfgs",
                "params": {
                    "maxstep": 0.1,
                    "alpha": 70,
                },
            },
        }
    )

    try:
        relaxed = optimize(atoms, config=optimizer_config, calculator=calc, verbose=True, constraints=constraints)
    except (OptimizerNotConvergedException, CalculationFailed) as e:
        logger.info(f"optimization failed, {e}")
        return None

    positions = relaxed.positions
    begin_id = 0
    new_mols = []
    for mol in mols:
        new_mol = mol.copy(keep_conformers=False)
        new_mol._conformers = [mol.conformers[conformer].copy()]
        new_mol.conformers[-1].coords = positions[begin_id : begin_id + mol.natoms]
        begin_id += mol.natoms
        new_mols.append(new_mol)

    return new_mols
