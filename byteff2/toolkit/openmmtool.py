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

import copy
import logging
import typing as T
from typing import Iterable

import networkx as nx
import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as openmm_unit
from openmm.app.gromacstopfile import GromacsTopFile

from bytemol.toolkit.asetool.basecalculator import BaseCalculator

logger = logging.getLogger(__name__)


def nx_covalent_map_and_pairs(natoms: int, pairs: Iterable) -> tuple[dict]:
    graph = nx.Graph()

    # filter redundant pairs
    pairs = set(tuple(sorted(list(_))) for _ in pairs)

    # ions are nodes with no edges
    graph.add_nodes_from([_ for _ in range(natoms)])
    graph.add_edges_from(pairs)
    assert graph.number_of_nodes() == natoms

    covalent_map = {}
    pair_recs = {'1-2': set(), '1-3': set(), '1-4': set(), '1-5': set(), '1-6': set()}
    for node in graph.nodes():
        neighbors = {'1-2': [], '1-3': [], '1-4': [], '1-5': [], '1-6': []}
        all_neighbors = set()
        for target_node, distance in nx.single_source_shortest_path_length(graph, source=node, cutoff=4).items():
            all_neighbors.add(target_node)
            if distance == 1:
                neighbors['1-2'].append(target_node)
                pair_recs['1-2'].add(tuple(sorted((node, target_node))))
            elif distance == 2:
                neighbors['1-3'].append(target_node)
                pair_recs['1-3'].add(tuple(sorted((node, target_node))))
            elif distance == 3:
                neighbors['1-4'].append(target_node)
                pair_recs['1-4'].add(tuple(sorted((node, target_node))))
            elif distance == 4:
                neighbors['1-5'].append(target_node)
                pair_recs['1-5'].add(tuple(sorted((node, target_node))))
        neighbors['1-6'] = list(set(range(natoms)) - {node} - all_neighbors)
        pair_recs['1-6'] |= {tuple(sorted((node, nb))) for nb in neighbors['1-6']}
        covalent_map[node] = neighbors

    return covalent_map, pair_recs


def generate_openmm_system(top_file,
                           nonbonded_params: dict,
                           unit_cell=None,
                           cutoff=1.0) -> tuple[GromacsTopFile, omm.System]:
    """ Build openmm system for ByteFF-Pol """

    assert isinstance(top_file, str) and top_file.endswith('.top'), 'input system must be a full gromacs topology'

    metadata = nonbonded_params.get('metadata', {})
    s12 = metadata.get('s12', 0.15)
    disp_damping = metadata.get('disp_damping', 0.4)

    logger.info(f's12: {s12}, disp_damping: {disp_damping}')
    if unit_cell:
        top = GromacsTopFile(top_file, unitCellDimensions=unit_cell)
        system: omm.System = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=False)
    else:
        top = GromacsTopFile(top_file)
        system: omm.System = top.createSystem(nonbondedMethod=app.NoCutoff, removeCMMotion=False)

    # remove nonbonded forces
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, omm.NonbondedForce):
            system.removeForce(i)
            break
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, omm.CustomNonbondedForce):
            system.removeForce(i)
            break
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, omm.CustomBondForce):
            system.removeForce(i)
            break

    # create nonbonded forces
    amoeba_force = omm.AmoebaMultipoleForce()
    lj14scale = 0.5
    lj15scale = 1.0
    func = f"6*A*exp(B*(1-r/rvdw))-C6/rvdw^6/({disp_damping}+(r/rvdw)^6)+({s12}/r)^12"
    comb_rule = "A=sqrt(A1*A2); B=sqrt(B1*B2); rvdw=(rvdw1+rvdw2)/2; C6=sqrt(C61*C62)"

    func += "-CTE*exp(-(CTL*r/rvdw)^3)/r^4"
    comb_rule += "; CTE=sqrt(CTE1*CTE2); CTL=sqrt(CTL1*CTL2)"

    print(func + " ; " + comb_rule)
    ljforce = omm.CustomNonbondedForce(func + " ; " + comb_rule)

    ljforce.addPerParticleParameter("A")  # epsilon
    ljforce.addPerParticleParameter("B")  # lambda
    ljforce.addPerParticleParameter("C6")  # C6
    ljforce.addPerParticleParameter("rvdw")  # rvdw
    ljforce.addPerParticleParameter("CTE")
    ljforce.addPerParticleParameter("CTL")

    lj14force = omm.CustomBondForce(f"S*6*A*exp(B*(1-r/rvdw))-S*C6/rvdw^6/({disp_damping}+(r/rvdw)^6)+S*({s12}/r)^12; ")
    lj14force.addPerBondParameter("A")  # epsilon
    lj14force.addPerBondParameter("B")  # lambda
    lj14force.addPerBondParameter("C6")  # C6
    lj14force.addPerBondParameter("rvdw")  # rvdw
    lj14force.addPerBondParameter("S")  # scale

    species = copy.deepcopy(top._molecules)
    logger.info(f'molecules {species}')
    residue_shift, index_shift = 0, 0
    residues = list(top.topology.residues())
    for mol_name, mol_number in species:
        params = nonbonded_params[mol_name]
        tot_charge = sum(params['charge'])
        assert abs(tot_charge - round(tot_charge)) < 1e-2

        # build covalent map (1-2, 1-3, 1-4, 1-5)
        residue = residues[residue_shift]
        residue_shift += mol_number
        natoms = len(list(residue.atoms()))
        bonds = [[b[0].index - index_shift, b[1].index - index_shift] for b in residue.bonds()]
        covalent_map, pairs = nx_covalent_map_and_pairs(natoms, bonds)

        for _ in range(mol_number):
            # handle each molecule
            for iatom in range(natoms):
                charge = params['charge'][iatom]
                alpha = params['alpha'][iatom]
                Rvdw = params['Rvdw'][iatom]

                lamb = params['lamb'][iatom]
                eps = params['eps'][iatom]
                C6 = params['C6'][iatom]
                pol_damp = params['pol_damping'][iatom]
                ct_eps = params['ct_eps'][iatom]
                ct_lamb = params['ct_lamb'][iatom]

                # amoeba_force

                dipoles = np.zeros(3)
                quadrupoles = np.zeros(9)
                axis_type = omm.AmoebaMultipoleForce.NoAxisType
                axis_indices = [0, 1, 2]
                thole = 0.39

                particle_idx = amoeba_force.addMultipole(
                    charge,
                    dipoles,  # permanent dipole is zero
                    quadrupoles,  # permanent quadrupole is zero 
                    axisType=axis_type,  # axisType
                    multipoleAtomZ=axis_indices[0],
                    multipoleAtomX=axis_indices[1],
                    multipoleAtomY=axis_indices[2],
                    thole=thole,
                    dampingFactor=(pol_damp)**(1 / 6),
                    polarity=alpha,
                )
                interactions = covalent_map[iatom]
                # colvalent map
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.Covalent12,
                                            [ii + index_shift for ii in interactions["1-2"]])
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.Covalent13,
                                            [ii + index_shift for ii in interactions["1-3"]])
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.Covalent14,
                                            [ii + index_shift for ii in interactions["1-4"]])
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.Covalent15,
                                            [ii + index_shift for ii in interactions["1-5"]])
                # polarization covalent map, each atom is an independent polarization group
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.PolarizationCovalent11,
                                            [particle_idx])
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.PolarizationCovalent12,
                                            [ii + index_shift for ii in interactions["1-2"]])
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.PolarizationCovalent13,
                                            [ii + index_shift for ii in interactions["1-3"]])
                amoeba_force.setCovalentMap(particle_idx, omm.AmoebaMultipoleForce.PolarizationCovalent14,
                                            [ii + index_shift for ii in interactions["1-4"]])

                # ljforce
                cc = [eps, lamb, C6, Rvdw]
                cc += [ct_eps, ct_lamb]
                ljforce.addParticle(cc)  # pylint: disable=used-before-assignment

            for p in ['1-2', '1-3', '1-4', '1-5']:
                for i, j in pairs[p]:
                    ljforce.addExclusion(i + index_shift, j + index_shift)

            for i, j in pairs['1-4']:
                Rvdw_i, Rvdw_j = params['Rvdw'][i], params['Rvdw'][j]
                lamb_i, lamb_j = params['lamb'][i], params['lamb'][j]
                eps_i, eps_j = params['eps'][i], params['eps'][j]
                C6_i, C6_j = params['C6'][i], params['C6'][j]
                lamb_ij = np.sqrt(lamb_i * lamb_j)
                C6_ij = np.sqrt(C6_i * C6_j)
                r_ij = 0.5 * (Rvdw_i + Rvdw_j)
                eps_ij = np.sqrt(eps_i * eps_j)
                lj14force.addBond(i + index_shift, j + index_shift, [eps_ij, lamb_ij, C6_ij, r_ij, lj14scale])

            for i, j in pairs['1-5']:
                Rvdw_i, Rvdw_j = params['Rvdw'][i], params['Rvdw'][j]
                lamb_i, lamb_j = params['lamb'][i], params['lamb'][j]
                eps_i, eps_j = params['eps'][i], params['eps'][j]
                C6_i, C6_j = params['C6'][i], params['C6'][j]
                lamb_ij = np.sqrt(lamb_i * lamb_j)
                C6_ij = np.sqrt(C6_i * C6_j)
                r_ij = 0.5 * (Rvdw_i + Rvdw_j)
                eps_ij = np.sqrt(eps_i * eps_j)
                lj14force.addBond(i + index_shift, j + index_shift, [eps_ij, lamb_ij, C6_ij, r_ij, lj15scale])

            index_shift += natoms

    if unit_cell is not None:
        logger.info('use PME')
        amoeba_force.setNonbondedMethod(omm.AmoebaMultipoleForce.PME)
        amoeba_force.setEwaldErrorTolerance(5.e-5)
        amoeba_force.setCutoffDistance(cutoff)  # nm
        ljforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
        ljforce.setCutoffDistance(cutoff)  # nm
        logger.info('use long range correction')
        ljforce.setUseLongRangeCorrection(True)
    else:
        amoeba_force.setNonbondedMethod(omm.AmoebaMultipoleForce.NoCutoff)
        ljforce.setNonbondedMethod(omm.CustomNonbondedForce.NoCutoff)

    amoeba_force.setPolarizationType(omm.AmoebaMultipoleForce.Mutual)
    amoeba_force.setMutualInducedMaxIterations(100)
    amoeba_force.setMutualInducedTargetEpsilon(1e-5)

    logger.info(f'amoeba force num multipoles {amoeba_force.getNumMultipoles()}')
    logger.info(f'amoeba force nonbonded method {amoeba_force.getNonbondedMethod()}')

    system.addForce(amoeba_force)
    system.addForce(ljforce)
    system.addForce(lj14force)

    return top, system


class AmoebaCalculator(BaseCalculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self,
                 top_file: str,
                 nonbonded_params: dict,
                 *,
                 platform_name: str = 'CPU',
                 separate_terms: bool = False,
                 apply_constraints: bool = False,
                 unit_cell=None,
                 cutoff=1.0):
        super().__init__()

        assert platform_name in ['CPU', 'Reference', 'CUDA']

        self.system: omm.System = None

        self.top, self.system = generate_openmm_system(top_file, nonbonded_params, unit_cell=unit_cell, cutoff=cutoff)

        self.integrator = omm.VerletIntegrator(0.001 * openmm_unit.picoseconds)
        self.platform = omm.Platform.getPlatformByName(platform_name)
        if platform_name == 'CUDA':
            self.platform.setPropertyDefaultValue('Precision', 'mixed')

        if separate_terms:
            self.forcegroups = {}
            self.separate_forces = {}
            self.separate_energy = {}
            for i in range(self.system.getNumForces()):
                force = self.system.getForce(i)
                force.setForceGroup(i)
                self.forcegroups[force] = (i, force.getName())

            logger.debug('recording separate energy contributions: %s', self.forcegroups)
        else:
            self.forcegroups = None
            self.separate_forces = None
            self.separate_energy = None

        self.simulation: app.Simulation = app.Simulation(self.top.topology, self.system, self.integrator, self.platform)

        self._apply_constraints: bool = apply_constraints
        self._last_state: omm.State = None
        self._last_positions: np.ndarray = None  # positions used in openmm calculation, with vsite/constraints applied
        self._n_atoms: int = self.system.getNumParticles()
        return

    def _calculate_without_restraint(self, coords: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
        """ Calculate force field energy and force in openmm.
        """

        assert coords.shape[0] == self._n_atoms
        self.simulation.context.setPositions(coords / 10)  # angstrom to nm

        if self.apply_constraints:
            self.simulation.context.applyConstraints(1e-5)

        # read result
        state = self.simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)  # pylint: disable=E1123
        self._last_state = state
        # convert nm to angstrom
        self._last_positions = np.array(state.getPositions().value_in_unit(openmm_unit.angstrom))
        # openmm uses kj/mol, nm. convert to kcal/mol, A
        energy = state.getPotentialEnergy().value_in_unit(openmm_unit.kilocalorie_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(openmm_unit.kilocalories_per_mole / openmm_unit.angstroms)

        return energy, forces

    def get_induced_dipole(self):
        # getInducedDipoles
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if isinstance(force, omm.AmoebaMultipoleForce):
                break
        dipoles = force.getInducedDipoles(self.simulation.context)
        dipoles = np.array([[dipole.x, dipole.y, dipole.z] for dipole in dipoles]) * 10  # in e * A
        return dipoles

    def get_separate_terms(self) -> T.Tuple[T.Dict, T.Dict]:
        '''return each energy & force terms in kcal, mol, A unit'''
        assert self.forcegroups is not None
        # openmm uses kj/mol, nm. convert to kcal/mol, A
        self.separate_energy = {}
        self.separate_forces = {}
        for _, (i, name) in self.forcegroups.items():
            state = self.simulation.context.getState(getEnergy=True, getForces=True, groups=2**i)  # pylint: disable=E1123
            energy = state.getPotentialEnergy().value_in_unit(openmm_unit.kilocalorie_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(openmm_unit.kilocalories_per_mole /
                                                                 openmm_unit.angstroms)
            if name in self.separate_forces:
                self.separate_energy[name] += energy
                self.separate_forces[name] += forces
            else:
                self.separate_energy[name] = energy
                self.separate_forces[name] = forces

        return self.separate_energy, self.separate_forces

    def serialize(self, xml_path: str) -> None:
        with open(xml_path, 'w') as output_file:
            xml_serialized_system = omm.XmlSerializer.serialize(self.system)
            output_file.write(xml_serialized_system)
        return

    @property
    def apply_constraints(self) -> bool:
        return self._apply_constraints

    @property
    def last_openmm_positions(self) -> np.ndarray:
        '''return actual openmm positions in A unit'''
        return self._last_positions

    @property
    def last_openmm_state(self) -> omm.State:
        return self._last_state
