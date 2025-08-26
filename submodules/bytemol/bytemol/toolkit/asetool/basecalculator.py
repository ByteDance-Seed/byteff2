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

import abc
import logging
import numbers
import os
from collections import namedtuple
from typing import Iterable, Tuple, Union

import ase
import numpy as np
from ase.calculators.calculator import Calculator

import bytemol.toolkit.asetool.geometry_numpy as kernel
from bytemol.core import Conformer, Molecule
from bytemol.core.conformer import is_energy_key, is_force_key
from bytemol.core.moleculegraph import MoleculeGraph
from bytemol.units import simple_unit as unit

Restraint = namedtuple("Restraint", ["atomidx", "force_constant", "target", "flat_bottom"])

logger = logging.getLogger(__name__)


class IncompatibleStructureError(Exception):
    """
    Raise when two structures are incompatible:
    i.e., mismatch of atom count, elements, etc.
    """


class HarmonicRestraints:

    def __init__(self):
        self.restraints = {
            "position": None,
            "dihedral": None,
        }

    def clear(self):
        for field in self.restraints:
            self.restraints[field] = None

    def set_restraints(self,
                       restraint_type: str,
                       atomidx: Iterable,
                       force_constant: Union[Iterable, float],
                       target: Iterable,
                       flat_bottom: float = 0.):
        '''
        for position restraint
        force_constant is in unit kcal/mol/A^2, target/flat_bottom are in unit A
        for dihedral restraint
        force_constant is in unit kcal/mol/degree^2, target is in unit degree, flat_bottom is ignored'''
        assert restraint_type in self.restraints, f"Restraint type {restraint_type} is illegal"
        assert self.restraints[restraint_type] is None, f"{restraint_type} restraints already exist"

        # check atomidx
        atomidx = np.array(atomidx, dtype=np.int64)
        if restraint_type == "position":
            assert atomidx.ndim == 1
        elif restraint_type == "dihedral":
            assert atomidx.ndim == 2 and atomidx.shape[1] == 4
        restraint_num = atomidx.shape[0]

        # check target
        target = np.array(target, dtype=np.float64)
        if restraint_type == "position":
            assert target.shape == (restraint_num, 3)
        else:
            assert target.shape == (restraint_num,)

        # check force_constant
        if np.isscalar(force_constant):
            force_constant = np.array([force_constant] * restraint_num, dtype=np.float64)
        else:
            force_constant = np.array(force_constant, dtype=np.float64)
        assert force_constant.shape == (restraint_num,)

        # check flat_bottom
        assert np.isscalar(flat_bottom)
        assert flat_bottom >= 0

        self.restraints[restraint_type] = Restraint(atomidx, force_constant, target, flat_bottom)

    def get_position_restraints_energy_forces(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' The potential of restraint at position r is
            E(r) = 0.5 * k_fb * ReLU(norm_2(r-r0) - r_fb) ^ 2
            where r0 is the reference position, r_fb is half width of flat-bottom,
            and k_fb is the force constant.
        '''
        atomidx, force_constant, target, flat_bottom = self.restraints["position"]  # pylint: disable=E0633
        displacement = coords[atomidx] - target
        distance = np.linalg.norm(displacement, axis=-1)
        delta_distance = distance - flat_bottom
        energy = np.sum(0.5 * force_constant * np.maximum(delta_distance, 0)**2)

        grad_displacement = np.where((delta_distance > 0)[..., np.newaxis], -force_constant[..., np.newaxis] *
                                     delta_distance[..., np.newaxis] * displacement / distance[..., np.newaxis], 0)

        forces = np.zeros_like(coords)
        forces[atomidx] = grad_displacement
        return energy, forces

    def get_dihedral_restraints_energy_forces(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' dihedral angle  
            E(phi) = 0.5 * k_fb * (|phi-phi0|-phi_fb)^2 if |phi-phi0| > phi_fb
            E(phi) = 0 otherwise
        '''
        atomidx, force_constant, target, flat_bottom = self.restraints["dihedral"]  # pylint: disable=E0633
        assert coords.ndim == 2
        assert coords.shape[-1] == 3
        energy = 0

        dihedral, f1, f2, f3, f4 = kernel.get_dihedral_angle_vec(coords[np.newaxis, np.newaxis, :, :],
                                                                 atomidx[np.newaxis, :, :])  # rad
        raw_delta_dihedral = (target[np.newaxis, np.newaxis, :] - unit.rad_to_degree(dihedral)) % 360.0
        # periodic, -pi and pi should generate zero restraint force
        delta_dihedral = np.where(raw_delta_dihedral > 180, 360 - raw_delta_dihedral, raw_delta_dihedral)
        delta_dihedral = np.where(delta_dihedral < flat_bottom, 0, delta_dihedral - flat_bottom)
        energy = np.sum(0.5 * force_constant * (delta_dihedral**2))

        # for debug
        # logger.info('%s %s %s', dihedral.shape, delta_dihedral.shape, raw_delta_dihedral.shape)

        # grad to force
        f1 = unit.rad_to_degree(f1)
        f2 = unit.rad_to_degree(f2)
        f3 = unit.rad_to_degree(f3)
        f4 = unit.rad_to_degree(f4)

        forces = np.zeros_like(coords)
        for ig in range(atomidx.shape[0]):
            if raw_delta_dihedral[0, 0, ig] > 180:
                sign = -1
            else:
                sign = 1
            forces[atomidx[ig][0], :] += sign * force_constant[ig] * delta_dihedral[0, 0, ig] * f1[0, 0, ig]
            forces[atomidx[ig][1], :] += sign * force_constant[ig] * delta_dihedral[0, 0, ig] * f2[0, 0, ig]
            forces[atomidx[ig][2], :] += sign * force_constant[ig] * delta_dihedral[0, 0, ig] * f3[0, 0, ig]
            forces[atomidx[ig][3], :] += sign * force_constant[ig] * delta_dihedral[0, 0, ig] * f4[0, 0, ig]

        return energy, forces

    def get_restraints_energy_and_forces(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        energy = 0
        forces = np.zeros_like(coords)
        if self.restraints["position"] is not None:
            res_energy, res_forces = self.get_position_restraints_energy_forces(coords)
            energy += res_energy
            forces += res_forces
        if self.restraints["dihedral"] is not None:
            res_energy, res_forces = self.get_dihedral_restraints_energy_forces(coords)
            energy += res_energy
            forces += res_forces
        return energy, forces


class BaseCalculator(Calculator, abc.ABC):
    implemented_properties = ["energy", "forces"]

    @abc.abstractmethod
    def _calculate_without_restraint(self,
                                     coords: np.ndarray,
                                     calc_force: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Delegate method to get results (e.g., energy, forces, etc.) without restraints.

        :param coords: atomic coordinates in A
        :param calc_force: if True, calculate forces
        :return: energy and forces in kcal/mol and kcal/mol/A, respectively        
        '''

    def _calculate_restraint(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Delegate method to get results (e.g., energy, forces, etc.) due to restraints only.
        
        :param coords: atomic coordinates in A
        :return: energy and forces in kcal/mol and kcal/mol/A, respectively        
        '''
        if self.restraints:
            res_energy, res_forces = self.restraints.get_restraints_energy_and_forces(coords)
            return res_energy, res_forces
        else:
            return 0, np.zeros_like(coords)

    def init_trajectory(self, trajectory: os.PathLike, template_mol: Molecule = None):
        """
        Initialize a `.xyz` file to log trajectory.

        :param trajectory: file path to a `.xyz` file
        :param template_mol: template Molecule object
        """
        if os.path.exists(trajectory):
            logger.warning(f"trajectory file {trajectory} exists, will overwrite it")
            with open(trajectory, "w"):
                pass
        if not trajectory.endswith(".xyz"):
            raise NameError("trajectory file should be .xyz format")
        self.trajectory = trajectory
        self.template_mol = template_mol.copy(keep_conformers=False) if isinstance(template_mol, Molecule) else None

    def _write_trajectory(self, atoms: ase.Atoms):
        new_unit_confdata = {}
        for key, val in self.results.items():
            if is_energy_key(key):
                new_unit_confdata[key] = unit.eV_to_kcal_mol(val)
            elif is_force_key(key):
                new_unit_confdata[key] = unit.eV_A_to_kcal_mol_A(val)
            else:
                new_unit_confdata[key] = val
        # we discard "energy" and "forces" keys to avoid misuse, as their values inlcude reatraints part
        new_unit_confdata.pop("energy")
        new_unit_confdata.pop("forces")
        conformer = Conformer(atoms.positions, atoms.symbols, confdata=new_unit_confdata)
        if self.template_mol is None:
            conformer.to_xyz(self.trajectory, append=True)
        else:
            self.template_mol._conformers = [conformer]
            self.template_mol.to_xyz(self.trajectory, append=True)

    def _assert_structure_compatibility(self, atoms: ase.Atoms):
        """"
        Check compatibility between input and internal structures.
        Raise exceptions or log warnings for incompatibility.

        Because derived Calculator classes (e.g., xTBCalculator, 
        GMXCalculator, etc.) have their own internal representation of a 
        structure, the implementation will be different. Common causes of
        incompability include: mismatch of atom count and elements for QM
        calculators, mismatch of topology for MM calculators, etc.

        TODO: If all the derived Calculator classes implement this method,
        we can decorate it with @abs.abstractmethod
        
        :param atoms: ase.Atoms object corresponding to a structure
        """

    def calculate(self, atoms: ase.Atoms, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Perform calculations by delegating method calls to 
        `_calculate_without_restraint()` and `_calculate_restraint()`.
        
        Results are stored in the internal `results` dictionary.

        Unlike base-class method `ase.calculators.calculator.Calculator.calculate`,
        `atoms` is not optional here. The topology of input structure (`atoms`) 
        should be compatible with the internal structure (`self.atoms`): atom count, 
        elements, etc.

        :param atoms: ase.Atoms object corresponding to a structure
        """
        self._assert_structure_compatibility(atoms)
        Calculator.calculate(self, atoms, *args, **kwargs)
        # convert coordinates
        coords = atoms.positions
        coords = np.asarray(coords)
        assert coords.ndim == 2
        assert coords.shape[-1] == 3

        energy, force = self._calculate_without_restraint(coords)
        res_energy, res_force = self._calculate_restraint(coords)

        # internally use kcal/mol, A
        # output to ase standard eV, A

        self.results['no_restraint_energy'] = unit.kcal_mol_to_eV(energy)
        self.results['no_restraint_forces'] = unit.kcal_mol_A_to_eV_A(force)
        self.results['restraints_energy'] = unit.kcal_mol_to_eV(res_energy)
        self.results['restraints_forces'] = unit.kcal_mol_A_to_eV_A(res_force)

        self.results['energy'] = unit.kcal_mol_to_eV(energy + res_energy)
        self.results['forces'] = unit.kcal_mol_A_to_eV_A(force + res_force)

        if self.trajectory is not None:
            self._write_trajectory(atoms)

    def __init__(self):
        super().__init__()
        self.restraints: HarmonicRestraints = None
        self.trajectory = None
        self.template_mol = None

    def set_position_restraints(self,
                                atomidx: Iterable,
                                force_constant: Union[Iterable, float],
                                target: Iterable[Iterable],
                                flat_bottom: float = 0.):
        """
        atomix: [npos]
        force_constant: [npos] or scalar, unit in kcal/mol/A^2
        target: [npos, 3], unit in Angstrom
        flat_bottom: scalar, unit in Angstrom
        """
        if self.restraints is None:
            self.restraints = HarmonicRestraints()

        self.reset()
        self.restraints.set_restraints("position", atomidx, force_constant, target, flat_bottom)

    def set_dihedral_restraints(
        self,
        atomidx: Iterable[Iterable],
        force_constant: Union[Iterable, float],
        target: Iterable[Iterable],
        flat_bottom: float = 0.,
    ):
        """
        atomix: [ndihedral, 4]
        force_constant: [ndihedral] or scalar, unit in kcal/mol/degree^2
        target: [ndihedral], unit in degree
        flat_bottom: scalar, unit in degree
        """
        if self.restraints is None:
            self.restraints = HarmonicRestraints()

        self.reset()
        self.restraints.set_restraints("dihedral", atomidx, force_constant, target, flat_bottom)

    def set_restraints(self,
                       molgraph: MoleculeGraph,
                       coords: list,
                       force_constant: float,
                       restraint_type: str = "rotatable_proper",
                       torsion_ids: list[int] = None,
                       flat_bottom: float = 0.):
        """
        molgraph: MoleculeGraph of molecule
        coords: coordinates of molecule
        force_constant: force constant of restraints, recommand 1 kcal/mol/A2 for position restraints and 0.1 kcal/mol/rad2 for dihedral restraints
        restraint_type:
            position: position restraints
            rotatable_proper: dihedral restraints on all rotatable propers
            tfd_proper: dihedral restranits on tfd propers
        torsion_ids: indices of constrained torsion atoms
        """
        if restraint_type == "position":
            restraint_torsions = []
            self.set_position_restraints(atomidx=np.arange(molgraph.natoms),
                                         force_constant=force_constant,
                                         target=coords,
                                         flat_bottom=flat_bottom)
        else:
            if restraint_type == "rotatable_proper":
                restraint_torsions = molgraph.get_rotatable_propers()
            elif restraint_type == "tfd_proper":
                restraint_torsions = molgraph.get_tfd_propers()
            if torsion_ids:
                if isinstance(torsion_ids[0], numbers.Number):
                    torsion_ids = [torsion_ids]

                exclude_torsions13 = [sorted(list(torsion_id[1:3])) for torsion_id in torsion_ids]
                restraint_torsions = [
                    res for res in restraint_torsions if sorted(list(res[1:3])) not in exclude_torsions13
                ]

        if restraint_torsions:
            restraints_angles = kernel.get_dihedral_angle_degree(coords, np.array(restraint_torsions))
            self.set_dihedral_restraints(
                atomidx=restraint_torsions,
                force_constant=force_constant,
                target=restraints_angles,
                flat_bottom=flat_bottom,
            )

    def clear_all_restraints(self):
        self.reset()
        self.restraints = None
