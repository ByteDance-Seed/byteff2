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
import os
from typing import Optional

import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as ou
import pandas as pd
from MDAnalysis.lib.formats.libdcd import DCDFile
from openmm.app.gromacstopfile import GromacsTopFile

from bytemol.utils import temporary_cd

logger = logging.getLogger(__name__)


def openmm_run(
    task_name: str,
    top: GromacsTopFile,
    system: omm.System,
    positions: list[omm.Vec3],
    integrator: omm.Integrator,
    reporter: app.StateDataReporter = None,
    work_dir: str = '.',
    minimize: bool = False,
    box_vec: Optional[omm.Vec3] = None,
    steps: int = None,
    temperature: float = 300.,
):

    with temporary_cd(work_dir):
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force_group = 1 if isinstance(force, (omm.AmoebaMultipoleForce, omm.NonbondedForce,
                                                  omm.CustomNonbondedForce)) else 0
            force.setForceGroup(force_group)
            # you should only see these in output:
            logger.info('system force %s, group %d', force.getName(), force.getForceGroup())

        platform = omm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'mixed')
        temperature = temperature * ou.kelvin  # Temperature for initial velocity
        sim = app.Simulation(top.topology, system, integrator, platform)
        sim.context.setPositions(positions)
        if box_vec is not None:
            sim.context.setPeriodicBoxVectors(*box_vec)
        if minimize:
            # Minimize the energy
            logger.info('Minimizing energy')
            sim.minimizeEnergy(
                maxIterations=1000,
                tolerance=10 * ou.kilojoules_per_mole / ou.nanometer,
            )
        # initialize temperature
        sim.context.setVelocitiesToTemperature(temperature)
        if reporter is not None:
            if isinstance(reporter, list):
                sim.reporters = reporter
            else:
                sim.reporters.append(reporter)

        # Run dynamics
        logger.info(f'Running {task_name}')
        sim.step(steps - sim.currentStep)
        logger.info(f'{task_name} done')
        # Get the state informations
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)  # pylint: disable=unexpected-keyword-arg
        positions = state.getPositions()  # nm
        box_vectors = state.getPeriodicBoxVectors()  # nm
    return positions, box_vectors


def npt_run(
    top: GromacsTopFile,
    system: omm.System,
    positions: list[omm.Vec3],
    npt_steps=2000000,
    temperature: float = 300,
    work_dir: str = '.',
):
    top = copy.deepcopy(top)
    system = copy.deepcopy(system)
    timestep = 2  # fs
    pressure = 1.0 * ou.atmospheres  # Target pressure
    frequency = 12  # Attempt volume change every 25 steps
    # default 4 ns
    barostat = omm.MonteCarloBarostat(pressure, temperature * ou.kelvin, frequency)
    system.addForce(barostat)
    integrator = omm.MTSLangevinIntegrator(temperature * ou.kelvin, 0.1 / ou.picosecond, timestep * ou.femtoseconds,
                                           [(0, 2), (1, 1)])
    state_reporter = app.StateDataReporter(
        file=os.path.join(work_dir, 'npt_state.csv'),
        reportInterval=500,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        progress=False,
        remainingTime=False,
        speed=True,
        elapsedTime=False,
        separator=',',
        systemMass=None,
        totalSteps=None,
        append=False,
    )
    dcd_reporter = app.DCDReporter(
        os.path.join(work_dir, 'npt.dcd'),
        reportInterval=500,
        enforcePeriodicBox=False,
    )
    return openmm_run(
        task_name='npt',
        top=top,
        system=system,
        positions=positions,
        integrator=integrator,
        reporter=[state_reporter, dcd_reporter],
        work_dir=work_dir,
        minimize=True,
        steps=npt_steps,
        temperature=temperature,
    )


def rescale_box(
    positions: list[omm.Vec3],
    box_vec: list[omm.Vec3],
    work_dir: str = None,
):
    # use average density
    csv_file = os.path.join(work_dir, 'npt_state.csv')
    box = pd.read_csv(csv_file)["Box Volume (nm^3)"]
    ave_length = np.mean(box[-500:])**(1 / 3)  # last 1 ns
    scale = ave_length / box_vec[0].x
    positions *= scale
    new_box_vec = []
    for vec in box_vec:
        new_box_vec.append(omm.Vec3(vec.x * scale, vec.y * scale, vec.z * scale) * ou.nanometers)
    logger.info('scale box by %.3f', scale)
    return positions, new_box_vec


def nvt_run(
    top: GromacsTopFile,
    system: omm.System,
    positions: list[omm.Vec3],
    box_vec: Optional[omm.Vec3],
    temperature: float,
    work_dir: str,
    nvt_steps: int,
    timestep: int = 2,  # fs
    extra_reporters: list | None = None,
):
    top = copy.deepcopy(top)
    system = copy.deepcopy(system)
    integrator = omm.MTSLangevinIntegrator(temperature * ou.kelvin, 0.1 / ou.picosecond, timestep * ou.femtoseconds,
                                           [(0, 2), (1, 1)])

    state_reporter = app.StateDataReporter(
        file=os.path.join(work_dir, 'nvt_state.csv'),
        reportInterval=500,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        progress=False,
        remainingTime=False,
        speed=True,
        elapsedTime=False,
        separator=',',
        systemMass=None,
        totalSteps=None,
        append=False,
    )
    dcd_reporter = app.DCDReporter(
        os.path.join(work_dir, 'nvt.dcd'),
        reportInterval=500,
        enforcePeriodicBox=False,
    )
    reporters = [state_reporter, dcd_reporter]
    if extra_reporters:
        reporters.extend(extra_reporters)
    return openmm_run(
        task_name='nvt',
        top=top,
        system=system,
        positions=positions,
        integrator=integrator,
        reporter=reporters,
        work_dir=work_dir,
        minimize=False,
        box_vec=box_vec,
        steps=nvt_steps,
        temperature=temperature,
    )


def volume_calc(work_dir):
    with temporary_cd(work_dir):
        csv_file = 'nvt_state.csv'
        result_df = pd.read_csv(csv_file)
        volume = result_df["Box Volume (nm^3)"].mean() * 1000
        temperature = result_df["Temperature (K)"].mean()
        return volume, temperature


def dcd_read(fp):
    position = []
    with DCDFile(fp) as dcd:
        # iterate over trajectory
        for frame in dcd:
            position.append(frame.xyz.copy())
    position = np.array(position)
    return position


class DipoleReporter:
    """
    Reporter for recording the total dipole moment of a system using the AMOEBA force field.
    This version considers only permanent charges (monopoles) and induced dipoles.
    
    Units:
    - Time: ps
    - Dipole components and magnitude: e*Angstrom
    """

    def __init__(self, file_path, reportInterval, system):
        """
        Initialize the reporter.
        
        Parameters
        ----------
        file_path : str
            Path to the output CSV file.
        reportInterval : int
            The interval (in steps) at which to write to the file.
        system : openmm.System
            The OpenMM System object to extract force and charge information.
        """
        self._reportInterval = int(reportInterval)
        self._file_path = file_path

        # 1. Locate the AmoebaMultipoleForce
        self._amoeba_force = None
        for i in range(system.getNumForces()):
            f = system.getForce(i)
            if isinstance(f, omm.AmoebaMultipoleForce):
                self._amoeba_force = f
                break

        if self._amoeba_force is None:
            raise RuntimeError("AmoebaMultipoleForce not found in the System.")

        # 2. Pre-extract permanent charges (monopoles)
        n_particles = system.getNumParticles()
        self._charges = np.zeros(n_particles)
        for i in range(n_particles):
            # params[0] is the charge q in units of elementary_charge
            params = self._amoeba_force.getMultipoleParameters(i)
            self._charges[i] = params[0].value_in_unit(ou.elementary_charge)

        # 3. Initialize the output file and write the header
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self._out = open(file_path, 'w')
        self._out.write('time_ps,Mx_eA,My_eA,Mz_eA,M_mag_eA\n')
        self._out.flush()

    def describeNextReport(self, simulation):
        """
        Describe the requirements for the next report.
        periodic=False is crucial: it requests unwrapped coordinates where 
        molecules are kept whole across periodic boundaries.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {'steps': steps, 'periodic': False, 'include': ['positions']}

    def report(self, simulation, state):
        """
        Calculate and record the dipole moment for the current state.
        """
        # 1. Get simulation time in ps
        t_ps = state.getTime().value_in_unit(ou.picoseconds)

        # 2. Get unwrapped positions in Angstroms
        # Because periodic=False, molecules are not torn by PBC
        pos = state.getPositions(asNumpy=True).value_in_unit(ou.angstrom)

        # 3. Calculate Monopole Contribution: sum(q_i * r_i)
        # Result in e * Angstrom
        m_monopole = np.sum(self._charges[:, np.newaxis] * pos, axis=0)

        # 4. Get Induced Dipoles from AMOEBA
        try:
            mu_ind_list = self._amoeba_force.getInducedDipoles(simulation.context)
        except omm.OpenMMException:
            # Re-initialize force reference if the Context has been updated/rebuilt
            self._reinit_force(simulation.system)
            mu_ind_list = self._amoeba_force.getInducedDipoles(simulation.context)

        # 5. Convert induced dipoles from e*nm to e*A and sum them up
        m_induced = np.array(mu_ind_list).sum(axis=0) * 10

        # 6. Total Dipole Vector
        m_total = m_monopole + m_induced
        m_mag = np.linalg.norm(m_total)

        # 7. Write to CSV file
        self._out.write(f"{t_ps:.4f},{m_total[0]:.6f},{m_total[1]:.6f},{m_total[2]:.6f},{m_mag:.6f}\n")
        self._out.flush()

    def _reinit_force(self, system):
        """Re-locate the AmoebaMultipoleForce instance in the System."""
        for i in range(system.getNumForces()):
            f = system.getForce(i)
            if isinstance(f, omm.AmoebaMultipoleForce):
                self._amoeba_force = f
                break

    def __del__(self):
        """Ensure the file is closed properly."""
        if hasattr(self, '_out'):
            self._out.close()
