"""
Polymer-specific MD simulation protocols.

This module provides simulation protocols tailored for polymer electrolyte systems,
including multi-stage equilibration and specialized transport analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import numpy as np
import logging

from byteff2.md_utils.md_run import npt_run, nvt_run
from byteff2.toolkit.protocol import TransportProtocol, Protocol

logger = logging.getLogger(__name__)


class EquilibrationStage:
    """Configuration for a single equilibration stage."""
    
    def __init__(
        self,
        name: str,
        ensemble: str,  # 'nvt' or 'npt'
        steps: int,
        temperature: float,
        pressure: Optional[float] = None,
        restraint_fc: Optional[float] = None,
        timestep: float = 0.001,  # ps
        **kwargs
    ):
        self.name = name
        self.ensemble = ensemble
        self.steps = steps
        self.temperature = temperature
        self.pressure = pressure if ensemble == 'npt' else None
        self.restraint_fc = restraint_fc
        self.timestep = timestep
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'ensemble': self.ensemble,
            'steps': self.steps,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'restraint_fc': self.restraint_fc,
            'timestep': self.timestep,
            **self.extra_params
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
        Run multi-stage equilibration.
        
        Parameters
        ----------
        topology_path : str
            Path to GROMACS topology file
        coordinates_path : str
            Path to initial coordinates (.gro file)
        mdp_template_dir : str
            Directory containing MDP templates
        
        Returns
        -------
        str
            Path to final equilibrated coordinates
        """
        current_coords = coordinates_path
        
        for i, stage in enumerate(self.stages):
            logger.info(f"Running equilibration stage {i+1}/{len(self.stages)}: {stage.name}")
            
            stage_dir = os.path.join(self.output_dir, f"stage_{i:02d}_{stage.name}")
            os.makedirs(stage_dir, exist_ok=True)
            
            if stage.ensemble == 'minimize':
                current_coords = self._run_minimization(
                    topology_path, current_coords, stage_dir, stage
                )
            elif stage.ensemble == 'nvt':
                current_coords = self._run_nvt_stage(
                    topology_path, current_coords, stage_dir, stage
                )
            elif stage.ensemble == 'npt':
                current_coords = self._run_npt_stage(
                    topology_path, current_coords, stage_dir, stage
                )
            
            logger.info(f"Completed stage {stage.name}")
        
        return current_coords
    
    def _run_minimization(
        self,
        topology_path: str,
        coords_path: str,
        output_dir: str,
        stage: EquilibrationStage
    ) -> str:
        """Run energy minimization stage."""
        from bytemol.toolkit.gmxtool import gmxscript
        
        mdp_content = self._generate_minimization_mdp(stage)
        mdp_path = os.path.join(output_dir, 'minimize.mdp')
        with open(mdp_path, 'w') as f:
            f.write(mdp_content)
        
        output_gro = os.path.join(output_dir, 'minimized.gro')
        
        script = gmxscript.GMXScript()
        script.grompp(
            mdp_path,
            coords_path,
            topology_path,
            os.path.join(output_dir, 'minimize.tpr')
        )
        script.mdrun(
            os.path.join(output_dir, 'minimize.tpr'),
            output_dir,
            'minimize'
        )
        script.run()
        
        return output_gro
    
    def _run_nvt_stage(
        self,
        topology_path: str,
        coords_path: str,
        output_dir: str,
        stage: EquilibrationStage
    ) -> str:
        """Run NVT equilibration stage."""
        mdp_content = self._generate_nvt_mdp(stage)
        mdp_path = os.path.join(output_dir, f'{stage.name}.mdp')
        with open(mdp_path, 'w') as f:
            f.write(mdp_content)
        
        # Generate restraint file if needed
        if stage.restraint_fc is not None:
            self._generate_restraint_file(coords_path, output_dir, stage.restraint_fc)
        
        output_gro = os.path.join(output_dir, f'{stage.name}.gro')
        
        nvt_run(
            topology_path,
            coords_path,
            mdp_path,
            output_dir,
            stage.name
        )
        
        return output_gro
    
    def _run_npt_stage(
        self,
        topology_path: str,
        coords_path: str,
        output_dir: str,
        stage: EquilibrationStage
    ) -> str:
        """Run NPT equilibration stage."""
        mdp_content = self._generate_npt_mdp(stage)
        mdp_path = os.path.join(output_dir, f'{stage.name}.mdp')
        with open(mdp_path, 'w') as f:
            f.write(mdp_content)
        
        output_gro = os.path.join(output_dir, f'{stage.name}.gro')
        
        npt_run(
            topology_path,
            coords_path,
            mdp_path,
            output_dir,
            stage.name
        )
        
        return output_gro
    
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
    
    def _generate_restraint_file(
        self,
        coords_path: str,
        output_dir: str,
        force_constant: float
    ) -> str:
        """Generate position restraint file for polymer backbone atoms."""
        # Parse GRO file to identify backbone atoms
        backbone_indices = self._identify_backbone_atoms(coords_path)
        
        restraint_path = os.path.join(output_dir, 'posre.itp')
        with open(restraint_path, 'w') as f:
            f.write("[ position_restraints ]\n")
            f.write("; atom  type      fx      fy      fz\n")
            for idx in backbone_indices:
                f.write(f"{idx:6d}     1  {force_constant:.1f}  {force_constant:.1f}  {force_constant:.1f}\n")
        
        return restraint_path
    
    def _identify_backbone_atoms(self, coords_path: str) -> List[int]:
        """Identify polymer backbone atom indices from coordinate file."""
        backbone_indices = []
        with open(coords_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header and footer
        for i, line in enumerate(lines[2:-1], start=1):
            if len(line) >= 15:
                atom_name = line[10:15].strip()
                # Identify backbone atoms (C, O in polymer backbone)
                if atom_name in ['C', 'C1', 'C2', 'O', 'O1', 'CA', 'CB']:
                    backbone_indices.append(i)
        
        return backbone_indices


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
        super().__init__(output_dir)
        self.params_dir = params_dir or os.path.join(output_dir, "params")
        self.polymer_chains = []
        self.coordination_data = {}
        self.config = {}
        
    # def __init__(self, params_dir: str, output_dir: str):
    #     super().__init__(params_dir, output_dir)
    #     self.polymer_chains = []
    #     self.coordination_data = {}
    
    def post_process(self):
        """Extended post-processing for polymer systems."""
        super().post_process()
        
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
            logger.warning("Could not find cations or coordinating atoms")
            return
        
        coordination_numbers = []
        cutoff = 3.0  # Angstrom, typical Li-O first shell
        
        for ts in u.trajectory[::10]:  # Sample every 10th frame
            dist_matrix = distances.distance_array(
                cations.positions,
                coord_atoms.positions,
                box=u.dimensions
            )
            cn_per_cation = np.sum(dist_matrix < cutoff, axis=1)
            coordination_numbers.append(cn_per_cation.mean())
        
        self.coordination_data = {
            'mean_coordination_number': np.mean(coordination_numbers),
            'std_coordination_number': np.std(coordination_numbers),
            'coordination_timeseries': coordination_numbers,
        }
        
        logger.info(f"Mean coordination number: {self.coordination_data['mean_coordination_number']:.2f}")
    
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
        ion_positions_rel = []  # Relative to local polymer
        
        for ts in u.trajectory:
            ion_positions_lab.append(cations.positions.copy())
            
            # Find local polymer environment for each ion
            # (simplified: use nearest polymer chain COM)
            # Full implementation would track specific coordination shells
        
        ion_positions_lab = np.array(ion_positions_lab)
        
        # Compute total MSD
        msd_total = np.mean(
            np.sum((ion_positions_lab[-1] - ion_positions_lab[0])**2, axis=1)
        )
        
        # Vehicular contribution (from chain diffusion)
        if hasattr(self, 'chain_diffusion'):
            t_total = len(u.trajectory) * u.trajectory.dt
            msd_vehicular = 6 * self.chain_diffusion * 1e5 * t_total  # Convert back
            
            self.diffusion_decomposition = {
                'total': msd_total / (6 * t_total) * 1e-5,
                'vehicular': self.chain_diffusion,
                'structural': (msd_total / (6 * t_total) * 1e-5) - self.chain_diffusion,
            }
            
            logger.info(f"Diffusion decomposition:")
            logger.info(f"  Total: {self.diffusion_decomposition['total']:.2e} cm²/s")
            logger.info(f"  Vehicular: {self.diffusion_decomposition['vehicular']:.2e} cm²/s")
            logger.info(f"  Structural: {self.diffusion_decomposition['structural']:.2e} cm²/s")
    
    def _compute_ion_hopping_statistics(self):
        """
        Compute ion hopping statistics between coordination sites.
        
        Identifies discrete hopping events and computes:
        - Hopping rate
        - Residence time distribution
        - Hopping distance distribution
        """
        logger.info("Computing ion hopping statistics...")
        
        # Implementation would track ion-polymer associations over time
        # and identify discrete hopping events
        
        self.hopping_statistics = {
            'mean_residence_time': None,  # ps
            'hopping_rate': None,  # per ps
            'mean_hopping_distance': None,  # Angstrom
        }
    
    def _save_polymer_analysis_results(self):
        """Save polymer-specific analysis results to files."""
        import json
        
        results = {
            'chain_diffusion_cm2_s': getattr(self, 'chain_diffusion', None),
            'coordination_data': getattr(self, 'coordination_data', {}),
            'diffusion_decomposition': getattr(self, 'diffusion_decomposition', {}),
            'hopping_statistics': getattr(self, 'hopping_statistics', {}),
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results = convert_numpy(results)
        
        output_path = os.path.join(self.output_dir, 'polymer_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Polymer analysis results saved to {output_path}")

    def run_full_workflow(self):
        """Run the complete polymer transport workflow."""
        logger.info("Running Polymer Transport Protocol")
        self.post_process()