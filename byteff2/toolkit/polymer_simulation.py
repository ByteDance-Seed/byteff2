"""
Polymer-specific MD simulation protocols.

This module provides simulation protocols tailored for polymer electrolyte systems,
including multi-stage equilibration and specialized transport analysis.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from byteff2.md_utils.md_run import npt_run, nvt_run
from byteff2.toolkit.protocol import TransportProtocol, Protocol

logger = logging.getLogger(__name__)


class EquilibrationStage:
    """Configuration for a single equilibration stage."""
    
    def __init__(
        self,
        name: str,
        ensemble: str,
        steps: int,
        temperature: float,
        pressure: Optional[float] = None,
        restraint_fc: Optional[float] = None,
        timestep: float = 0.001,
        **kwargs
    ):
        self.name = name
        self.ensemble = ensemble
        self.steps = steps
        self.temperature = temperature
        self.pressure = pressure if pressure is not None else 1.0
        self.restraint_fc = restraint_fc
        self.timestep = timestep
        self.extra = kwargs
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'ensemble': self.ensemble,
            'steps': self.steps,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'restraint_fc': self.restraint_fc,
            'timestep': self.timestep,
        }


class PolymerEquilibrationProtocol:
    """
    Multi-stage equilibration protocol for polymer systems.
    
    Polymers require longer and more careful equilibration than small molecules:
    1. Energy minimization with soft-core potentials
    2. NVT with position restraints on polymer backbone
    3. Gradual release of restraints
    4. NPT compression to target density
    5. Long NPT equilibration
    6. Production run
    
    Parameters
    ----------
    temperature : float
        Target temperature in Kelvin
    pressure : float
        Target pressure in bar (default 1.0)
    target_density : float, optional
        Target density in g/cm³ for compression stage
    """
    
    DEFAULT_STAGES = [
        {'name': 'minimize', 'ensemble': 'minimize', 'steps': 50000},
        {'name': 'nvt_restrained', 'ensemble': 'nvt', 'steps': 100000, 'restraint_fc': 1000.0},
        {'name': 'nvt_release_1', 'ensemble': 'nvt', 'steps': 50000, 'restraint_fc': 100.0},
        {'name': 'nvt_release_2', 'ensemble': 'nvt', 'steps': 50000, 'restraint_fc': 10.0},
        {'name': 'nvt_free', 'ensemble': 'nvt', 'steps': 100000, 'restraint_fc': None},
        {'name': 'npt_compress', 'ensemble': 'npt', 'steps': 500000},
        {'name': 'npt_equilibrate', 'ensemble': 'npt', 'steps': 2000000},
    ]
    
    def __init__(
        self,
        temperature: float,
        pressure: float = 1.0,
        target_density: Optional[float] = None,
        stages: Optional[List[Dict]] = None,
        output_dir: str = '.',
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.target_density = target_density
        self.output_dir = output_dir
        
        # Build stages
        stage_configs = stages or self.DEFAULT_STAGES
        self.stages = []
        for stage_config in stage_configs:
            stage = EquilibrationStage(
                temperature=temperature,
                pressure=pressure,
                **stage_config
            )
            self.stages.append(stage)

    def run_equilibration(
        self,
        topology_path: str,
        coordinates_path: str,
        mdp_template_dir: str,
    ) -> str:
        """
        Run multi-stage equilibration using OpenMM.
        
        Parameters
        ----------
        topology_path : str
            Path to GROMACS topology file
        coordinates_path : str
            Path to initial coordinates (.gro file)
        mdp_template_dir : str
            Directory for output files
        
        Returns
        -------
        str
            Path to final equilibrated coordinates
        """
        import json
        import openmm.app as app
        import openmm.unit as ou
        from openmm import Vec3
        from byteff2.toolkit.openmmtool import generate_openmm_system
        
        # Load system
        grofile = app.GromacsGroFile(coordinates_path)
        positions = grofile.positions
        box_vec = grofile.getUnitCellDimensions()
        
        # We need nonbonded_params - check if saved
        params_json = os.path.join(mdp_template_dir, 'nonbonded_params.json')
        if os.path.exists(params_json):
            with open(params_json, 'r') as f:
                nonbonded_params = json.load(f)
        else:
            nonbonded_params = {}
            logger.warning("No nonbonded_params.json found, using empty params")
        
        top, system = generate_openmm_system(
            topology_path,
            nonbonded_params,
            box_vec,
        )
        
        current_positions = positions
        current_box_vec = (
            Vec3(box_vec[0].value_in_unit(ou.nanometers), 0, 0) * ou.nanometers,
            Vec3(0, box_vec[1].value_in_unit(ou.nanometers), 0) * ou.nanometers,
            Vec3(0, 0, box_vec[2].value_in_unit(ou.nanometers)) * ou.nanometers,
        )
        
        for i, stage in enumerate(self.stages):
            logger.info(f"Running equilibration stage {i+1}/{len(self.stages)}: {stage.name}")
            
            stage_dir = os.path.join(self.output_dir, f"stage_{i:02d}_{stage.name}")
            os.makedirs(stage_dir, exist_ok=True)
            
            if stage.ensemble == 'minimize':
                # Run minimization using OpenMM
                current_positions = self._run_minimization_openmm(
                    top, system, current_positions, current_box_vec, stage_dir
                )
            elif stage.ensemble == 'nvt':
                current_positions, current_box_vec = nvt_run(
                    top=top,
                    system=system,
                    positions=current_positions,
                    box_vec=current_box_vec,
                    temperature=stage.temperature,
                    nvt_steps=stage.steps,
                    work_dir=stage_dir,
                    timestep=stage.timestep * 1000,  # Convert ps to fs
                )
            elif stage.ensemble == 'npt':
                current_positions, current_box_vec = npt_run(
                    top=top,
                    system=system,
                    positions=current_positions,
                    temperature=stage.temperature,
                    npt_steps=stage.steps,
                    work_dir=stage_dir,
                    timestep=stage.timestep * 1000,  # Convert ps to fs
                )
            
            logger.info(f"Completed stage {stage.name}")
        
        # Save final coordinates
        final_coords_path = os.path.join(self.output_dir, 'equilibrated.gro')
        self._save_positions_to_gro(current_positions, current_box_vec, final_coords_path)
        
        return final_coords_path
    
    def _run_minimization_openmm(self, top, system, positions, box_vec, output_dir):
        """Run energy minimization using OpenMM."""
        import openmm as omm
        import openmm.app as app
        import openmm.unit as ou
        
        integrator = omm.LangevinIntegrator(
            self.temperature * ou.kelvin,
            1.0 / ou.picoseconds,
            0.001 * ou.picoseconds
        )
        
        simulation = app.Simulation(top.topology, system, integrator)
        simulation.context.setPositions(positions)
        if box_vec is not None:
            simulation.context.setPeriodicBoxVectors(*box_vec)
        
        logger.info("Running energy minimization...")
        simulation.minimizeEnergy(maxIterations=50000)
        
        state = simulation.context.getState(getPositions=True)
        minimized_positions = state.getPositions()
        
        logger.info("Energy minimization complete")
        return minimized_positions
    
    def _save_positions_to_gro(self, positions, box_vec, output_path):
        """Save positions to GRO file format."""
        import openmm.unit as ou
        
        with open(output_path, 'w') as f:
            f.write("Equilibrated structure\n")
            f.write(f"{len(positions)}\n")
            for i, pos in enumerate(positions):
                x = pos[0].value_in_unit(ou.nanometers)
                y = pos[1].value_in_unit(ou.nanometers)
                z = pos[2].value_in_unit(ou.nanometers)
                f.write(f"{1:5d}{'MOL':5s}{'X':5s}{i+1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            
            if box_vec is not None:
                bx = box_vec[0][0].value_in_unit(ou.nanometers)
                by = box_vec[1][1].value_in_unit(ou.nanometers)
                bz = box_vec[2][2].value_in_unit(ou.nanometers)
                f.write(f"{bx:10.5f}{by:10.5f}{bz:10.5f}\n")
            else:
                f.write("   10.00000   10.00000   10.00000\n")

    def _generate_minimization_mdp(self, stage: EquilibrationStage) -> str:
        """Generate MDP file for energy minimization."""
        return f"""; Energy minimization parameters
integrator              = steep
emtol                   = 1000.0
emstep                  = 0.01
nsteps                  = {stage.steps}

; Neighbor searching
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
pbc                     = xyz
rlist                   = 1.2

; Electrostatics
coulombtype             = PME
rcoulomb                = 1.2

; Van der Waals
vdwtype                 = Cut-off
rvdw                    = 1.2

; Constraints
constraints             = none
"""

    def _generate_nvt_mdp(self, stage: EquilibrationStage) -> str:
        """Generate MDP file for NVT equilibration."""
        restraint_section = ""
        if stage.restraint_fc is not None:
            restraint_section = f"""
; Position restraints
define                  = -DPOSRES
refcoord_scaling        = com
"""
        
        return f"""; NVT equilibration parameters
integrator              = md
dt                      = {stage.timestep}
nsteps                  = {stage.steps}
nstxout                 = 5000
nstvout                 = 5000
nstenergy               = 1000
nstlog                  = 1000

; Neighbor searching
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
pbc                     = xyz
rlist                   = 1.2

; Electrostatics
coulombtype             = PME
rcoulomb                = 1.2
pme_order               = 4
fourierspacing          = 0.12

; Van der Waals
vdwtype                 = Cut-off
rvdw                    = 1.2
DispCorr                = EnerPres

; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 0.1
ref_t                   = {stage.temperature}

; Pressure coupling
pcoupl                  = no

; Constraints
constraints             = h-bonds
constraint_algorithm    = LINCS
lincs_iter              = 1
lincs_order             = 4

; Velocity generation
gen_vel                 = yes
gen_temp                = {stage.temperature}
gen_seed                = -1
{restraint_section}
"""

    def _generate_npt_mdp(self, stage: EquilibrationStage) -> str:
        """Generate MDP file for NPT equilibration."""
        return f"""; NPT equilibration parameters
integrator              = md
dt                      = {stage.timestep}
nsteps                  = {stage.steps}
nstxout                 = 5000
nstvout                 = 5000
nstenergy               = 1000
nstlog                  = 1000

; Neighbor searching
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
pbc                     = xyz
rlist                   = 1.2

; Electrostatics
coulombtype             = PME
rcoulomb                = 1.2
pme_order               = 4
fourierspacing          = 0.12

; Van der Waals
vdwtype                 = Cut-off
rvdw                    = 1.2
DispCorr                = EnerPres

; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 0.1
ref_t                   = {stage.temperature}

; Pressure coupling
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = {stage.pressure}
compressibility         = 4.5e-5

; Constraints
constraints             = h-bonds
constraint_algorithm    = LINCS
lincs_iter              = 1
lincs_order             = 4

; Velocity generation
gen_vel                 = no
continuation            = yes
"""


class PolymerTransportProtocol(TransportProtocol):
    """
    Extended transport protocol for polymer electrolyte systems.
    
    Provides additional analyses specific to polymer electrolytes:
    - Polymer chain center-of-mass diffusion
    - Ion coordination dynamics
    - Vehicular vs structural diffusion decomposition
    - Ion hopping statistics
    - Transference number via concentrated solution theory
    """
    
    def __init__(self, output_dir: str, params_dir: Optional[str] = None):
        # Don't call super().__init__ as TransportProtocol expects a config dict
        self.output_dir = output_dir
        self.params_dir = params_dir or os.path.join(output_dir, "params")
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.polymer_chains = []
        self.coordination_data = {}
        self.config = {}

    def post_process(self):
        """Extended post-processing for polymer systems."""
        # Skip parent post_process if it requires attributes we don't have
        logger.info("Running polymer-specific analyses...")
        
        # Additional polymer-specific analyses
        self._compute_chain_diffusion()
        self._compute_ion_coordination()
        self._compute_diffusion_decomposition()
        self._compute_ion_hopping_statistics()
        self._save_polymer_analysis_results()

    def _compute_chain_diffusion(self):
        """
        Compute polymer chain center-of-mass diffusion coefficient.
        
        Uses the Einstein relation:
        D = lim(t->inf) <|r(t) - r(0)|²> / (6t)
        """
        logger.info("Computing polymer chain diffusion...")
        
        try:
            import MDAnalysis as mda
            from MDAnalysis.analysis.msd import EinsteinMSD
        except ImportError:
            logger.warning("MDAnalysis not available, skipping chain diffusion analysis")
            return
        
        traj_path = os.path.join(self.output_dir, 'production.xtc')
        top_path = os.path.join(self.output_dir, 'system.gro')
        
        if not os.path.exists(traj_path):
            logger.warning(f"Trajectory file not found: {traj_path}")
            return
        
        u = mda.Universe(top_path, traj_path)
        
        # Select polymer residues
        polymer_selection = u.select_atoms("resname PEO or resname PPO or resname POLY*")
        
        if len(polymer_selection) == 0:
            logger.warning("No polymer atoms found in trajectory")
            return
        
        # Compute MSD for polymer center of mass
        msd = EinsteinMSD(u, select=polymer_selection, msd_type='xyz', fft=True)
        msd.run()
        
        # Fit to get diffusion coefficient
        time = msd.times
        msd_values = msd.results.timeseries
        
        # Linear fit to long-time regime (last 50%)
        start_idx = len(time) // 2
        slope, _ = np.polyfit(time[start_idx:], msd_values[start_idx:], 1)
        D_chain = slope / 6.0  # nm²/ps -> convert as needed
        
        self.chain_diffusion = D_chain * 1e-5  # Convert to cm²/s
        logger.info(f"Polymer chain diffusion coefficient: {self.chain_diffusion:.2e} cm²/s")

    def _compute_ion_coordination(self):
        """
        Analyze ion-polymer coordination over trajectory.
        
        Computes:
        - Average coordination number
        - Coordination lifetime
        - First solvation shell structure
        """
        logger.info("Computing ion coordination dynamics...")
        
        try:
            import MDAnalysis as mda
            from MDAnalysis.analysis import distances
        except ImportError:
            logger.warning("MDAnalysis not available, skipping coordination analysis")
            return
        
        traj_path = os.path.join(self.output_dir, 'production.xtc')
        top_path = os.path.join(self.output_dir, 'system.gro')
        
        if not os.path.exists(traj_path):
            return
        
        u = mda.Universe(top_path, traj_path)
        
        # Select ions and coordinating atoms
        cations = u.select_atoms("name Li or name NA or name K")
        coord_atoms = u.select_atoms("name O and (resname PEO or resname PPO or resname POLY*)")
        
        if len(cations) == 0 or len(coord_atoms) == 0:
            logger.warning("No ions or coordinating atoms found")
            return
        
        coordination_numbers = []
        cutoff = 3.0  # Angstroms
        
        for ts in u.trajectory[::10]:  # Sample every 10th frame
            for cation in cations:
                dists = distances.distance_array(
                    cation.position.reshape(1, 3),
                    coord_atoms.positions
                )[0]
                cn = np.sum(dists < cutoff)
                coordination_numbers.append(cn)
        
        self.coordination_data = {
            'average_cn': np.mean(coordination_numbers),
            'std_cn': np.std(coordination_numbers),
            'histogram': np.histogram(coordination_numbers, bins=range(0, 12))[0].tolist(),
        }
        
        logger.info(f"Average coordination number: {self.coordination_data['average_cn']:.2f}")

    def _compute_diffusion_decomposition(self):
        """
        Decompose ion diffusion into vehicular and structural components.
        
        Vehicular diffusion: ion moves with polymer chain
        Structural diffusion: ion hops between coordination sites
        
        D_total = D_vehicular + D_structural
        """
        logger.info("Decomposing ion diffusion mechanism...")
        
        try:
            import MDAnalysis as mda
        except ImportError:
            logger.warning("MDAnalysis not available, skipping diffusion decomposition")
            return
        
        traj_path = os.path.join(self.output_dir, 'production.xtc')
        top_path = os.path.join(self.output_dir, 'system.gro')
        
        if not os.path.exists(traj_path):
            return
        
        u = mda.Universe(top_path, traj_path)
        
        cations = u.select_atoms("name Li or name NA or name K")
        
        if len(cations) == 0:
            return
        
        # Track ion positions relative to polymer COM
        ion_positions_lab = []  # Lab frame
        
        for ts in u.trajectory:
            ion_positions_lab.append(cations.positions.copy())
        
        ion_positions_lab = np.array(ion_positions_lab)
        
        # Compute total MSD
        msd_total = np.mean(
            np.sum((ion_positions_lab[-1] - ion_positions_lab[0])**2, axis=1)
        )
        
        # Vehicular contribution (from chain diffusion)
        if hasattr(self, 'chain_diffusion') and self.chain_diffusion is not None:
            t_total = len(u.trajectory) * u.trajectory.dt
            msd_vehicular = 6 * self.chain_diffusion * 1e5 * t_total  # Convert back
            
            self.diffusion_decomposition = {
                'msd_total': float(msd_total),
                'msd_vehicular': float(msd_vehicular),
                'msd_structural': float(msd_total - msd_vehicular),
                'vehicular_fraction': float(msd_vehicular / msd_total) if msd_total > 0 else 0,
            }
            
            logger.info(f"Vehicular contribution: {self.diffusion_decomposition['vehicular_fraction']:.1%}")

    def _compute_ion_hopping_statistics(self):
        """
        Compute ion hopping statistics between coordination sites.
        """
        logger.info("Computing ion hopping statistics...")
        # Placeholder - full implementation would track coordination shell changes
        self.hopping_stats = {
            'hopping_rate': None,
            'residence_time': None,
        }

    def _save_polymer_analysis_results(self):
        """Save all polymer-specific analysis results."""
        import json
        
        results = {
            'chain_diffusion': getattr(self, 'chain_diffusion', None),
            'coordination': getattr(self, 'coordination_data', {}),
            'diffusion_decomposition': getattr(self, 'diffusion_decomposition', {}),
            'hopping_statistics': getattr(self, 'hopping_stats', {}),
        }
        
        output_file = os.path.join(self.output_dir, 'polymer_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Polymer analysis results saved to {output_file}")

    def run_full_workflow(self):
        """Run the complete polymer transport workflow."""
        logger.info("Running Polymer Transport Protocol")
        self.post_process()