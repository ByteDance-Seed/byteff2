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

import struct
import shutil
import tempfile

logger = logging.getLogger(__name__)

## added by me on 01-19-2026
def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate that a checkpoint file is complete and readable."""
    if not os.path.exists(checkpoint_path):
        return False
    try:
        # Check file size is reasonable (not truncated)
        file_size = os.path.getsize(checkpoint_path)
        if file_size < 1000:  # Minimum reasonable size
            return False
        
        # Try to read the checkpoint header
        with open(checkpoint_path, 'rb') as f:
            # OpenMM checkpoints start with a version number
            header = f.read(4)
            if len(header) < 4:
                return False
        return True
    except Exception:
        return False

def validate_positions(positions) -> bool:
    """Check if any positions contain NaN or Inf values."""
    import numpy as np
    pos_array = np.array([[p.x, p.y, p.z] for p in positions])
    return not (np.any(np.isnan(pos_array)) or np.any(np.isinf(pos_array)))

class SafeCheckpointReporter(app.CheckpointReporter):
    """Checkpoint reporter that writes atomically to prevent corruption."""
    
    def __init__(self, file, reportInterval):
        super().__init__(file, reportInterval)
        self._final_path = file
    
    def report(self, simulation, state):
        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(self._final_path) or '.',
            suffix='.chk.tmp'
        )
        os.close(temp_fd)
        try:
            simulation.saveCheckpoint(temp_path)
            # Atomic rename (on POSIX systems)
            shutil.move(temp_path, self._final_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

def recover_from_trajectory(
    dcd_path: str,
    csv_path: str,
    backoff_frames: int = 2,
    max_backoff: int = 50,
) -> tuple:
    """
    Recover valid positions from trajectory with progressive backoff.
    
    Returns (positions, box_length, frame_index) or raises if unrecoverable.
    """
    import numpy as np
    from MDAnalysis.coordinates.DCD import DCDFile
    import pandas as pd
    
    if not os.path.exists(dcd_path):
        raise FileNotFoundError(f"DCD file not found: {dcd_path}")
    
    # Read all frames
    positions_list = []
    with DCDFile(dcd_path) as dcd:
        for frame in dcd:
            positions_list.append(frame.xyz.copy())
    
    if len(positions_list) == 0:
        raise ValueError("DCD file contains no frames")
    
    # Read box volumes
    df = pd.read_csv(csv_path)
    box_volumes = df["Box Volume (nm^3)"].values
    
    # Try progressively earlier frames until we find valid positions
    for backoff in range(backoff_frames, min(max_backoff, len(positions_list))):
        frame_idx = len(positions_list) - 1 - backoff
        if frame_idx < 0:
            continue
            
        pos = positions_list[frame_idx]
        
        # Validate positions
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            logger.warning(f"Frame {frame_idx} contains NaN/Inf, trying earlier frame")
            continue
        
        # Check for unreasonable positions (atoms too far apart or overlapping)
        if np.any(np.abs(pos) > 1000):  # nm, unreasonably large
            logger.warning(f"Frame {frame_idx} has unreasonable positions, trying earlier")
            continue
        
        # Get corresponding box size
        csv_idx = min(frame_idx, len(box_volumes) - 1)
        box_length = box_volumes[csv_idx] ** (1/3)
        
        logger.info(f"Recovered valid positions from frame {frame_idx}")
        return pos, box_length, frame_idx
    
    raise ValueError(f"Could not find valid positions in last {max_backoff} frames")

def stabilize_polarizable_system(
    simulation: app.Simulation,
    system: omm.System,
    temperature: float,
    max_iterations: int = 5,
):
    """
    Stabilize a polarizable system after position recovery.
    """
    # Find and temporarily modify polarization settings
    amoeba_force = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, omm.AmoebaMultipoleForce):
            amoeba_force = force
            break
    
    if amoeba_force is not None:
        # Store original mutual induced settings
        original_max_iter = amoeba_force.getMutualInducedMaxIterations()
        original_target_epsilon = amoeba_force.getMutualInducedTargetEpsilon()
        
        try:
            # Use tighter convergence during stabilization
            amoeba_force.setMutualInducedMaxIterations(500)
            amoeba_force.setMutualInducedTargetEpsilon(1e-7)
            simulation.context.reinitialize(preserveState=True)
            
            # Gentle minimization with position restraints conceptually
            for i in range(max_iterations):
                try:
                    simulation.minimizeEnergy(maxIterations=100, tolerance=10.0)
                    # Check if stable
                    state = simulation.context.getState(getEnergy=True)
                    pe = state.getPotentialEnergy().value_in_unit(ou.kilojoules_per_mole)
                    if not (np.isnan(pe) or np.isinf(pe)):
                        logger.info(f"Stabilization iteration {i}: PE = {pe:.2f} kJ/mol")
                        break
                except Exception as e:
                    logger.warning(f"Stabilization iteration {i} failed: {e}")
                    # Reassign velocities and try again
                    simulation.context.setVelocitiesToTemperature(temperature * 0.5)
        finally:
            # Restore original settings
            amoeba_force.setMutualInducedMaxIterations(original_max_iter)
            amoeba_force.setMutualInducedTargetEpsilon(original_target_epsilon)
            simulation.context.reinitialize(preserveState=True)
    
    # Final velocity assignment at target temperature
    simulation.context.setVelocitiesToTemperature(temperature)
###

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
    resume_max_backoff_frames: int = 50,  # NEW: maximum frames to search back (01-19-2026)
    resume_validate_checkpoint: bool = True,  # NEW: validate before loading (01-19-2026)
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
        
        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            # NEW: Validate checkpoint before attempting to load: 01-19-2026
            checkpoint_valid = (
                validate_checkpoint(checkpoint_path) 
                if resume_validate_checkpoint else True
            )
            
            if checkpoint_valid:
                try:
                    sim.loadCheckpoint(checkpoint_path)
                    # Validate loaded positions
                    state = sim.context.getState(getPositions=True)
                    if not validate_positions(state.getPositions()):
                        raise ValueError("Loaded checkpoint contains NaN positions")
                    logger.info("Successfully resumed from checkpoint")
                except Exception as e:
                    logger.warning(f"Checkpoint load failed: {e}, attempting trajectory recovery")
                    checkpoint_valid = False
            
            if not checkpoint_valid:
                # Enhanced trajectory recovery
                dcd_path = dcd_path_override or f'{task_name}.dcd'
                csv_path = state_csv_override or f'{task_name}_state.csv'
                
                try:
                    recovered_pos, box_length, frame_idx = recover_from_trajectory(
                        dcd_path, csv_path,
                        backoff_frames=resume_safe_backoff_frames,
                        max_backoff=resume_max_backoff_frames,
                    )
                    
                    # Apply recovered state
                    sim.context.setPositions(recovered_pos * ou.angstroms)
                    sim.context.setPeriodicBoxVectors(
                        omm.Vec3(box_length, 0, 0) * ou.nanometers,
                        omm.Vec3(0, box_length, 0) * ou.nanometers,
                        omm.Vec3(0, 0, box_length) * ou.nanometers,
                    )
                    
                    # Stabilize polarizable system
                    stabilize_polarizable_system(sim, system, temperature)
                    
                    logger.info(f"Recovered from trajectory frame {frame_idx}")
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
                    raise
        ###

        # # Resume from checkpoint if requested and available
        # if resume and checkpoint_path and os.path.isfile(checkpoint_path):
        #     logger.info('Resuming %s from checkpoint %s', task_name, checkpoint_path)
        #     sim.loadCheckpoint(checkpoint_path)
        #     minimize = False  # do not minimize when resuming

        if minimize:
            # Minimize the energy
            logger.info('Minimizing energy')
            sim.minimizeEnergy(
                maxIterations=1000,
                tolerance=10 * ou.kilojoules_per_mole / ou.nanometer,
            )
        # initialize temperature only when not resuming from a checkpoint
        # if not (resume and checkpoint_path and os.path.isfile(checkpoint_path)):
        if not (resume and checkpoint_path and os.path.exists(checkpoint_path)):
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
    traj_interval: int = 500,
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
        reportInterval=traj_interval,
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
            reportInterval=traj_interval,
            enforcePeriodicBox=False,
            append=append_logs,
        )
    except TypeError:
        dcd_reporter = app.DCDReporter(
            dcd_path,
            reportInterval=traj_interval,
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
        traj_interval: int = 500,
):
    top = copy.deepcopy(top)
    system = copy.deepcopy(system)
    integrator = omm.MTSLangevinIntegrator(temperature * ou.kelvin, 0.1 / ou.picosecond, timestep * ou.femtoseconds,
                                           [(0, 2), (1, 1)])

    append_logs = bool(resume and os.path.isfile(os.path.join(work_dir, 'nvt.chk')))
    state_reporter = app.StateDataReporter(
        file='nvt_state.csv',
        reportInterval=traj_interval,
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
            reportInterval=traj_interval,
            enforcePeriodicBox=False,
            append=append_logs,
        )
    except TypeError:
        dcd_reporter = app.DCDReporter(
            dcd_path,
            reportInterval=traj_interval,
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



class DipoleReporter:
    """
    Reporter for recording the total dipole moment of a system using the AMOEBA force field.
    This version considers only permanent charges (monopoles) and induced dipoles.
    
    Units:
    - Time: ps
    - Dipole components and magnitude: e*Angstrom
    """

    def __init__(self, file_path, reportInterval, system, append: bool = False):
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
        append : bool
            If True, append to an existing file instead of overwriting it.
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
        mode = 'a' if append and os.path.isfile(file_path) else 'w'
        self._out = open(file_path, mode)
        if mode == 'w':
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
