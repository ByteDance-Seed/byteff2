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
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    dcd_path_override: Optional[str] = None,
    state_csv_override: Optional[str] = None,
    resume_safe_backoff_frames: int = 2,
    resume_safe_minimize: bool = True,
    resume_safe_warmup_steps: int = 5000,
    resume_safe_warmup_step_factor: float = 2.0,
    resume_safe_disable_barostat_warmup: bool = True,
):

    with temporary_cd(work_dir):
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force_group = 1 if isinstance(force, (omm.AmoebaMultipoleForce, omm.NonbondedForce,
                                                  omm.CustomNonbondedForce)) else 0
            force.setForceGroup(force_group)
            # you should only see these in output:
            logger.info('system force %s, group %d', force.getName(), force.getForceGroup())
        
        # Select OpenMM platform with optional env override
        # Env overrides (first match): BYTEFF2_OPENMM_PLATFORM, OPENMM_PLATFORM, OPENMM_DEFAULT_PLATFORM
        # Precision override: BYTEFF2_OPENMM_PRECISION or OPENMM_PRECISION (CUDA/OpenCL only)
        try:
            from simtk.openmm import Platform  # noqa: WPS433
        except Exception:
            Platform = omm.Platform
        requested = os.environ.get('BYTEFF2_OPENMM_PLATFORM') or os.environ.get('OPENMM_PLATFORM') or os.environ.get(
            'OPENMM_DEFAULT_PLATFORM')
        precision = os.environ.get('BYTEFF2_OPENMM_PRECISION') or os.environ.get('OPENMM_PRECISION') or 'mixed'
        try:
            if requested:
                platform = Platform.getPlatformByName(requested)
            else:
                available = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
                print("Available platforms:", available)
                if "CUDA" in available:
                    platform = Platform.getPlatformByName("CUDA")
                elif "OpenCL" in available:
                    platform = Platform.getPlatformByName("OpenCL")
                elif "CPU" in available:
                    platform = Platform.getPlatformByName("CPU")
                else:
                    platform = Platform.getPlatformByName("Reference")
            # Set precision if supported
            if platform.getName() in ("CUDA", "OpenCL"):
                try:
                    platform.setPropertyDefaultValue('Precision', precision)
                except Exception:
                    pass
        except Exception:
            # final fallback
            platform = omm.Platform.getPlatformByName('CPU')
        temperature = temperature * ou.kelvin  # Temperature for initial velocity
        sim = app.Simulation(top.topology, system, integrator, platform)
        sim.context.setPositions(positions)
        if box_vec is not None:
            sim.context.setPeriodicBoxVectors(*box_vec)
        # Resume from checkpoint if requested and available
        if resume and checkpoint_path and os.path.isfile(checkpoint_path):
            logger.info('Resuming %s from checkpoint %s', task_name, checkpoint_path)
            sim.loadCheckpoint(checkpoint_path)
            minimize = False  # do not minimize when resuming

        if minimize:
            # Minimize the energy
            logger.info('Minimizing energy')
            sim.minimizeEnergy(
                maxIterations=1000,
                tolerance=10 * ou.kilojoules_per_mole / ou.nanometer,
            )
        # initialize temperature only when not resuming from a checkpoint
        if not (resume and checkpoint_path and os.path.isfile(checkpoint_path)):
            sim.context.setVelocitiesToTemperature(temperature)
        if reporter is not None:
            if isinstance(reporter, list):
                sim.reporters = reporter
            else:
                sim.reporters.append(reporter)

        # Run dynamics
        to_run = int(steps) - int(sim.currentStep)
        if to_run < 0:
            logger.info(f'{task_name}: target steps (%d) already reached (current=%d); skipping run', steps, sim.currentStep)
            to_run = 0
        logger.info(f'Running {task_name}')
        if to_run:
            try:
                sim.step(to_run)
            except Exception as e:
                msg = str(e)
                is_nan = ('NaN' in msg or 'nan' in msg)
                if not (resume and is_nan):
                    raise
                logger.warning('%s encountered NaN while resuming; attempting safe fallback to last stable trajectory frame', task_name)
                # Determine artifacts
                dcd_path = dcd_path_override or f'{task_name}.dcd'
                csv_path = state_csv_override or f'{task_name}_state.csv'
                if not os.path.isabs(dcd_path):
                    dcd_path = os.path.join(os.getcwd(), dcd_path)
                if not os.path.isabs(csv_path):
                    csv_path = os.path.join(os.getcwd(), csv_path)
                # Load positions
                try:
                    frames = dcd_read(dcd_path)
                except Exception:
                    frames = np.array([])
                if frames is None or len(frames) == 0:
                    logger.error('Safe-resume failed: could not read frames from %s', dcd_path)
                    raise
                idx = max(0, len(frames) - 1 - int(resume_safe_backoff_frames or 0))
                last = frames[idx]
                last_positions = [omm.Vec3(x, y, z) * ou.nanometers for x, y, z in last]
                # Try to set box from CSV
                try:
                    df = pd.read_csv(csv_path)
                    if 'Box Volume (nm^3)' in df.columns and len(df) > 0:
                        # choose corresponding or last row
                        ridx = min(idx, len(df) - 1)
                        L = float(df['Box Volume (nm^3)'].iloc[ridx]) ** (1.0 / 3.0)
                        sim.context.setPeriodicBoxVectors(
                            omm.Vec3(L, 0.0, 0.0) * ou.nanometers,
                            omm.Vec3(0.0, L, 0.0) * ou.nanometers,
                            omm.Vec3(0.0, 0.0, L) * ou.nanometers,
                        )
                except Exception:
                    pass
                # Apply positions, reset velocities
                sim.context.setPositions(last_positions)
                sim.context.setVelocitiesToTemperature(temperature)
                if resume_safe_minimize:
                    try:
                        sim.minimizeEnergy(maxIterations=200)
                    except Exception:
                        pass
                # Optional warmup: disable barostat and reduce step size temporarily
                try:
                    # capture original settings
                    orig_step = None
                    try:
                        orig_step = integrator.getStepSize()
                    except Exception:
                        pass
                    barostat = None
                    orig_freq = None
                    for i in range(system.getNumForces()):
                        f = system.getForce(i)
                        if isinstance(f, omm.MonteCarloBarostat):
                            barostat = f
                            try:
                                orig_freq = f.getFrequency()
                            except Exception:
                                orig_freq = None
                            break
                    if resume_safe_disable_barostat_warmup and barostat is not None:
                        try:
                            barostat.setFrequency(0)
                            sim.context.reinitialize(preserveState=True)
                        except Exception:
                            pass
                    if resume_safe_warmup_steps and resume_safe_warmup_steps > 0 and orig_step is not None:
                        try:
                            warm_step = float(orig_step.value_in_unit(ou.femtoseconds)) / float(max(resume_safe_warmup_step_factor, 1.0))
                            integrator.setStepSize(warm_step * ou.femtoseconds)
                        except Exception:
                            pass
                        try:
                            sim.step(int(resume_safe_warmup_steps))
                        except Exception:
                            # if warmup fails, proceed to attempting main run with original settings
                            pass
                    # restore settings
                    if orig_step is not None:
                        try:
                            integrator.setStepSize(orig_step)
                        except Exception:
                            pass
                    if resume_safe_disable_barostat_warmup and barostat is not None and orig_freq is not None:
                        try:
                            barostat.setFrequency(orig_freq)
                            sim.context.reinitialize(preserveState=True)
                        except Exception:
                            pass
                except Exception:
                    pass
                # continue
                to_run2 = int(steps) - int(sim.currentStep)
                if to_run2 > 0:
                    sim.step(to_run2)
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
    resume: bool = False,
    checkpoint_interval: int = 5000,
    timestep: int = 2,  # fs
    state_csv_override: Optional[str] = None,
    dcd_path_override: Optional[str] = None,
    resume_safe_backoff_frames: int = 2,
    resume_safe_minimize: bool = True,
):
    top = copy.deepcopy(top)
    system = copy.deepcopy(system)
    pressure = 1.0 * ou.atmospheres  # Target pressure
    frequency = 12  # Attempt volume change every 25 steps
    # default 4 ns
    barostat = omm.MonteCarloBarostat(pressure, temperature * ou.kelvin, frequency)
    system.addForce(barostat)
    integrator = omm.MTSLangevinIntegrator(temperature * ou.kelvin, 0.1 / ou.picosecond, timestep * ou.femtoseconds,
                                           [(0, 2), (1, 1)])
    append_logs = bool(resume and os.path.isfile(os.path.join(work_dir, 'npt.chk')))
    state_reporter = app.StateDataReporter(
        file='npt_state.csv',
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
        append=append_logs,
    )
    dcd_path = dcd_path_override or 'npt.dcd'
    try:
        dcd_reporter = app.DCDReporter(
            dcd_path,
            reportInterval=500,
            enforcePeriodicBox=False,
            append=append_logs,
        )
    except TypeError:
        dcd_reporter = app.DCDReporter(
            dcd_path,
            reportInterval=500,
            enforcePeriodicBox=False,
        )
    reporters = [state_reporter, dcd_reporter]
    if checkpoint_interval and checkpoint_interval > 0:
        reporters.append(app.CheckpointReporter('npt.chk', checkpoint_interval))
    return openmm_run(
        task_name='npt',
        top=top,
        system=system,
        positions=positions,
        integrator=integrator,
        reporter=reporters,
        work_dir=work_dir,
        minimize=True,
        steps=npt_steps,
        temperature=temperature,
        resume=resume,
        checkpoint_path='npt.chk',
        dcd_path_override=dcd_path,
        state_csv_override=state_csv_override,
        resume_safe_backoff_frames=resume_safe_backoff_frames,
        resume_safe_minimize=resume_safe_minimize,
    )


def rescale_box(
    positions: list[omm.Vec3],
    box_vec,
    work_dir: str = None,
    csv_override: str = None,
):
    """
    Rescale positions and box vectors using the average NPT box length.

    Accepts box_vec in multiple forms:
    - tuple/list of three Vec3 (OpenMM periodic box vectors)
    - a single Vec3 of lengths (Lx, Ly, Lz)
    - a tuple/list of three floats (Lx, Ly, Lz) in nm
    """
    # use average density
    csv_file = csv_override if csv_override else os.path.join(work_dir, 'npt_state.csv')
    box = pd.read_csv(csv_file)["Box Volume (nm^3)"]
    ave_length = np.mean(box[-500:]) ** (1 / 3)  # last 1 ns

    # Normalize input box specification robustly
    def _to_numeric_triplet(bv):
        # None -> unknown; caller will handle
        if bv is None:
            return None
        # Quantity wrapping Vec3
        try:
            if hasattr(bv, 'value_in_unit'):
                tmp = bv.value_in_unit(ou.nanometer)
                # value_in_unit may return a Vec3
                if hasattr(tmp, 'x') and hasattr(tmp, 'y') and hasattr(tmp, 'z'):
                    return (float(tmp.x), float(tmp.y), float(tmp.z))
                # or a scalar/sequence
                bv = tmp
        except Exception:
            pass
        # Vec3
        if hasattr(bv, 'x') and hasattr(bv, 'y') and hasattr(bv, 'z'):
            return (float(bv.x), float(bv.y), float(bv.z))
        # tuple/list of three Vec3 -> use vector norms (handles triclinic)
        if isinstance(bv, (list, tuple)) and len(bv) == 3 and all(hasattr(x, 'x') for x in bv):
            import math
            def vlen(v):
                return math.sqrt(float(v.x)**2 + float(v.y)**2 + float(v.z)**2)
            return (vlen(bv[0]), vlen(bv[1]), vlen(bv[2]))
        # tuple/list of three numbers/quantities
        if isinstance(bv, (list, tuple)) and len(bv) == 3:
            vals = []
            for x in bv:
                if hasattr(x, 'value_in_unit'):
                    try:
                        x = x.value_in_unit(ou.nanometer)
                    except Exception:
                        x = float(x)
                vals.append(float(x))
            return (vals[0], vals[1], vals[2])
        return None

    triplet = _to_numeric_triplet(box_vec)
    if triplet is None:
        # Fall back: if we can't determine current box, assume current box length equals target average length
        Lx = Ly = Lz = float(ave_length)
        scale = 1.0
    else:
        Lx, Ly, Lz = triplet
        scale = ave_length / Lx if Lx else 1.0

    positions *= scale

    new_box_vec = [
        omm.Vec3(Lx * scale, 0.0, 0.0) * ou.nanometers,
        omm.Vec3(0.0, Ly * scale, 0.0) * ou.nanometers,
        omm.Vec3(0.0, 0.0, Lz * scale) * ou.nanometers,
    ]

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
        resume: bool = False,
        checkpoint_interval: int = 5000,
        state_csv_override: Optional[str] = None,
        dcd_path_override: Optional[str] = None,
        resume_safe_backoff_frames: int = 2,
        resume_safe_minimize: bool = True,
):
    top = copy.deepcopy(top)
    system = copy.deepcopy(system)
    integrator = omm.MTSLangevinIntegrator(temperature * ou.kelvin, 0.1 / ou.picosecond, timestep * ou.femtoseconds,
                                           [(0, 2), (1, 1)])

    append_logs = bool(resume and os.path.isfile(os.path.join(work_dir, 'nvt.chk')))
    state_reporter = app.StateDataReporter(
        file='nvt_state.csv',
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
        append=append_logs,
    )
    dcd_path = dcd_path_override or 'nvt.dcd'
    try:
        dcd_reporter = app.DCDReporter(
            dcd_path,
            reportInterval=500,
            enforcePeriodicBox=False,
            append=append_logs,
        )
    except TypeError:
        dcd_reporter = app.DCDReporter(
            dcd_path,
            reportInterval=500,
            enforcePeriodicBox=False,
        )
    reporters = [state_reporter, dcd_reporter]
    if checkpoint_interval and checkpoint_interval > 0:
        reporters.append(app.CheckpointReporter('nvt.chk', checkpoint_interval))
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
        resume=resume,
        checkpoint_path='nvt.chk',
        dcd_path_override=dcd_path,
        state_csv_override=state_csv_override,
        resume_safe_backoff_frames=resume_safe_backoff_frames,
        resume_safe_minimize=resume_safe_minimize,
    )


def volume_calc(work_dir, csv_override: str = None):
    with temporary_cd(work_dir):
        candidates = []
        if csv_override:
            candidates.append(csv_override)
        candidates.extend(['nvt_state.csv', 'nvt_results.csv', 'nvt.csv'])
        csv_file = None
        for cand in candidates:
            if cand and os.path.isfile(cand):
                csv_file = cand
                break
        if not csv_file:
            raise FileNotFoundError(f'Could not find any NVT state CSV among: {candidates} in {os.getcwd()}')
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
