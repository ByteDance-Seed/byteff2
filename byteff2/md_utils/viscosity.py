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

# The ViscosityReporter is modified from https://github.com/z-gong/openmm-velocityVerlet/blob/master/examples/ommhelper/reporter/viscosityreporter.py
# Portions copyright (c) 2020 the Authors.

# Authors: Zheng Gong

# Contributors:

# Part of the TGNH code comes from [scychon's openmm_drudeNose](https://github.com/scychon/openmm_drudeNose).

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

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

    def __init__(self, file, reportInterval):
        self._reportInterval = reportInterval
        self._out = open(file, 'w')
        self._hasInitialized = False

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
        return {'steps': steps, 'periodic': False, 'include': []}

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
):
    from velocityverletplugin import VVIntegrator
    timestep = 1  # fs, MTS is not supported in VVIntegrator
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
    vis_reporter = ViscosityReporter('viscosity.csv', 50)
    return openmm_run(
        task_name='nonequ',
        top=top,
        system=system,
        positions=positions,
        integrator=integrator,
        reporter=[vis_reporter],
        work_dir=work_dir,
        minimize=False,
        box_vec=box_vec,
        steps=nonequ_steps,
        temperature=temperature,
    )


def viscosity_calc(work_dir):
    with temporary_cd(work_dir):
        csv_file = 'viscosity.csv'
        viscosity = pd.read_csv(csv_file, sep='\t')["1/Viscosity (1/Pa.s)"]
        assert len(viscosity) >= 10000, 'viscosity trajectory too short'
        vis_1 = viscosity[1000:]  # skip first 50 ps
        return 1 / vis_1.mean() * 1000
