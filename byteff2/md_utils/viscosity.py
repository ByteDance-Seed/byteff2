# Copyright (c) 2020 Zheng Gong
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.08.25
#
# Original file was released under MIT, with the full license text
# available at https://github.com/z-gong/openmm-velocityVerlet/blob/master/.
#
# This modified file is released under the same license.

# The ViscosityReporter is modified from https://github.com/z-gong/openmm-velocityVerlet/blob/master/examples/ommhelper/reporter/viscosityreporter.py

import os
from typing import Optional

import openmm as omm
import openmm.app as app
import openmm.unit as ou
import pandas as pd
from openmm.app.gromacstopfile import GromacsTopFile

from bytemol.utils import temporary_cd

from .md_run import openmm_run


class ViscosityReporter(object):
    '''
    ViscosityReporter report the viscosity using cosine periodic perturbation method.
    A integrator supporting this method is required.
    e.g. the VVIntegrator from https://github.com/z-gong/openmm-velocityVerlet.

    Parameters
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    '''

    def __init__(self, file, reportInterval, append: bool = False):
        self._reportInterval = reportInterval
        mode = 'a' if append else 'w'
        self._out = open(file, mode)
        self._hasInitialized = False
        # If appending and file already has content, skip header
        if append and os.path.isfile(file) and os.path.getsize(file) > 0:
            self._hasInitialized = True

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        # Return 6-tuple format: (steps, needPositions, needVelocities,
        # needForces, needEnergies, periodic). The dict form
        # {'steps':..., 'periodic':..., 'include':[]} only works on OpenMM
        # builds that normalize it; OpenMM 8.1.2's _simulate indexes by [0]
        # and raises KeyError: 0 on a dict.
        return (steps, False, False, False, False, False)

    def report(self, simulation: app.Simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._hasInitialized = True
            print('#"Step"\t"Acceleration (nm/ps^2)"\t"VelocityAmplitude (nm/ps)"\t"1/Viscosity (1/Pa.s)"',
                  file=self._out)

        ps = ou.picosecond
        nm = ou.nanometer

        acceleration = simulation.integrator.getCosAcceleration().value_in_unit(nm / ps**2)
        vMax, invVis = simulation.integrator.getViscosity()
        vMax = vMax.value_in_unit(nm / ps)
        invVis = invVis.value_in_unit((ou.pascal * ou.second)**-1)
        print(simulation.currentStep, acceleration, vMax, invVis, sep='\t', file=self._out)

        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()


def nonequ_run(
    top: GromacsTopFile,
    system: omm.System,
    positions: list[omm.Vec3],
    box_vec: Optional[omm.Vec3],
    temperature: float,
    work_dir: str,
    nonequ_steps: int,
    resume: bool = False,
    checkpoint_interval: int = 5000,
    timestep_fs: int = 1,
):
    from velocityverletplugin import VVIntegrator
    # VVIntegrator supports variable step size; default 1 fs
    timestep = int(timestep_fs)
    integrator = VVIntegrator(
        temperature=temperature * ou.kelvin,
        frequency=1.0 / ou.picoseconds,
        drudeTemperature=temperature * ou.kelvin,  # placeholder, useless
        drudeFrequency=100 / ou.picoseconds,  # placeholder, useless
        stepSize=timestep * ou.femtoseconds,  # necessary
        numNHChains=3,  # default
        loopsPerStep=3,  # default
    )
    integrator.setUseMiddleScheme(True)
    integrator.setCosAcceleration(0.02)  # in openmm standard unit nm/ps^2
    append_logs = bool(resume and os.path.isfile(os.path.join(work_dir, 'nonequ.chk')))
    vis_reporter = ViscosityReporter('viscosity.csv', 50, append=append_logs)
    chk_reporter = app.CheckpointReporter('nonequ.chk', checkpoint_interval)
    return openmm_run(
        task_name='nonequ',
        top=top,
        system=system,
        positions=positions,
        integrator=integrator,
        reporter=[vis_reporter, chk_reporter],
        work_dir=work_dir,
        minimize=False,
        box_vec=box_vec,
        steps=nonequ_steps,
        temperature=temperature,
        resume=resume,
        checkpoint_path='nonequ.chk',
    )


def viscosity_calc(work_dir):
    with temporary_cd(work_dir):
        csv_file = 'viscosity.csv'
        viscosity = pd.read_csv(csv_file, sep='\t')["1/Viscosity (1/Pa.s)"]
        assert len(viscosity) >= 10000, 'viscosity trajectory too short'
        vis_1 = viscosity[1000:]  # skip first 50 ps
        return 1 / vis_1.mean() * 1000
