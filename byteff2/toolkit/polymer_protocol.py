# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Polymer electrolyte protocol classes for ByteFF2.

This module extends the base Protocol classes to support polymer electrolyte systems,
including polymer chain building, specialized box construction, and polymer-specific
simulation protocols.
"""

import os
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from byteff2.toolkit.protocol import (
    Protocol, 
    Component, 
    ComponentType, 
    TransportProtocol,
    DensityProtocol,
    load_topo,
    predict_box,
    predict_density,
)

logger = logging.getLogger(__name__)


class PolymerType(Enum):
    """Enumeration of polymer architecture types."""
    LINEAR = "linear"
    BRANCHED = "branched"
    CROSSLINKED = "crosslinked"
    BLOCK_COPOLYMER = "block_copolymer"
    RANDOM_COPOLYMER = "random_copolymer"


class BoxBuilderType(Enum):
    """Enumeration of available box building backends."""
    GROMACS = "gromacs"
    PACKMOL = "packmol"
    AMORPHOUS = "amorphous"


@dataclass
class PolymerComponent(Component):
    """
    Extended Component class for polymer systems.
    
    Attributes:
        polymer_type: Type of polymer architecture (linear, branched, etc.)
        degree_of_polymerization: Number of repeat units in the chain
        monomer_smiles: SMILES string of the monomer unit with connection points [*]
        end_groups: Tuple of (left_cap, right_cap) SMILES for chain termination
        tacticity: Stereochemistry preference (isotactic, syndiotactic, atactic)
    """
    polymer_type: PolymerType = PolymerType.LINEAR
    degree_of_polymerization: int = 1
    monomer_smiles: Optional[str] = None
    end_groups: Tuple[Optional[str], Optional[str]] = (None, None)
    tacticity: str = "atactic"
    
    @classmethod
    def from_config(cls, name: str, config: dict, smiles: str) -> 'PolymerComponent':
        """
        Create a PolymerComponent from configuration dictionary.
        
        Args:
            name: Component name
            config: Configuration dictionary with polymer-specific fields
            smiles: Full polymer SMILES (or monomer SMILES if building)
            
        Returns:
            PolymerComponent instance
        """
        comp_type = config.get("type", "polymer")
        if comp_type == "polymer":
            component_type = ComponentType.SOLVENT  # Polymers treated as matrix
        elif comp_type == "salt_cation":
            component_type = ComponentType.CATION
        elif comp_type == "salt_anion":
            component_type = ComponentType.ANION
        else:
            component_type = ComponentType.SOLVENT
            
        return cls(
            name=name,
            smiles=smiles,
            type=component_type,
            molar_num=config.get("count", 1),
            polymer_type=PolymerType(config.get("polymer_type", "linear")),
            degree_of_polymerization=config.get("dp", config.get("degree_of_polymerization", 1)),
            monomer_smiles=config.get("monomer_smiles"),
            end_groups=(config.get("left_cap"), config.get("right_cap")),
            tacticity=config.get("tacticity", "atactic"),
        )


class PolymerElectrolyteProtocol(Protocol):
    """
    Base protocol for polymer electrolyte MD simulations.
    
    This protocol handles:
    - Polymer chain generation from monomer SMILES
    - Box building using Packmol (for polymer compatibility)
    - Force field parameter generation with fragment-based approach for large molecules
    - Multi-stage equilibration for proper polymer relaxation
    """
    
    # def __init__(self, output_dir: str, params_dir: Optional[str] = None):
    #     super().__init__(output_dir)
    #     self.params_dir = params_dir or os.path.join(output_dir, "params")
    #     self.polymer_components: Dict[str, PolymerComponent] = {}
    #     self.box_builder_type = BoxBuilderType.PACKMOL
    #     self._polymer_builder = None
    
    def __init__(self, output_dir: str, params_dir: Optional[str] = None):
        # Base Protocol expects (params_dir, output_dir) in that order
        resolved_params_dir = params_dir or os.path.join(output_dir, "params")
        super().__init__(resolved_params_dir, output_dir)
        self.config = {}
        self.polymer_components: Dict[str, PolymerComponent] = {}
        self.box_builder_type = BoxBuilderType.PACKMOL
        self._polymer_builder = None
        
    def setup_from_config(self, config: dict):
        """
        Setup protocol from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing components, smiles, etc.
        """
        self.config = config
        self.box_builder_type = BoxBuilderType(
            config.get("box_builder", "packmol")
        )
        
        # Parse components
        components = config.get("components", {})
        smiles_dict = config.get("smiles", {})
        
        for name, comp_config in components.items():
            if isinstance(comp_config, dict):
                smiles = smiles_dict.get(name, comp_config.get("smiles", ""))
                if comp_config.get("type") == "polymer":
                    self.polymer_components[name] = PolymerComponent.from_config(
                        name, comp_config, smiles
                    )
                    
    def get_box_builder(self):
        """Get the appropriate box builder based on configuration."""
        from byteff2.toolkit.box_builders import (
            GMXBoxBuilder,
            PackmolBoxBuilder,
            AmorphousBoxBuilder,
        )
        
        builders = {
            BoxBuilderType.GROMACS: GMXBoxBuilder,
            BoxBuilderType.PACKMOL: PackmolBoxBuilder,
            BoxBuilderType.AMORPHOUS: AmorphousBoxBuilder,
        }
        
        builder_class = builders.get(self.box_builder_type, PackmolBoxBuilder)
        return builder_class()
    
    def build_polymer_chains(self, output_dir: str) -> Dict[str, str]:
        """
        Build polymer chains from monomer specifications.
        
        Args:
            output_dir: Directory to save generated polymer structures
            
        Returns:
            Dictionary mapping polymer names to generated structure file paths
        """
        from byteff2.toolkit.polymer_builder import PolymerChainBuilder
        
        os.makedirs(output_dir, exist_ok=True)
        generated_structures = {}
        
        for name, polymer in self.polymer_components.items():
            if polymer.monomer_smiles:
                builder = PolymerChainBuilder(
                    monomer_smiles=polymer.monomer_smiles,
                    dp=polymer.degree_of_polymerization,
                    end_group_left=polymer.end_groups[0],
                    end_group_right=polymer.end_groups[1],
                    tacticity=polymer.tacticity,
                )
                
                # Build chain and save
                mol = builder.build_chain()
                output_path = os.path.join(output_dir, f"{name}.pdb")
                builder.save_pdb(mol, output_path, resname=name[:3].upper())
                generated_structures[name] = output_path
                
                # Update SMILES to full polymer SMILES
                polymer.smiles = builder.get_polymer_smiles()
                
                logger.info(f"Built polymer chain {name} with DP={polymer.degree_of_polymerization}")
                
        return generated_structures
    
    def generate_ff_params_polymer(self, component_smiles: dict, force: bool = False):
        """
        Generate force field parameters for polymer systems.
        
        For large polymers (>500 atoms), uses fragment-based parameterization.
        
        Args:
            component_smiles: Dictionary of component names to SMILES strings
            force: Force regeneration even if parameters exist
            
        Returns:
            Dictionary of non-bonded parameters for each component
        """
        from rdkit import Chem
        from byteff2.model import load_model
        from byteff2.utils import get_data_file_path
        
        model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
        model = load_model(os.path.dirname(model_dir))
        all_nb_params = {}
        
        max_atoms_direct = self.config.get("max_atoms_direct_param", 500)
        
        for mol_name, smiles in component_smiles.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not parse SMILES for {mol_name}: {smiles}")
                continue
                
            num_atoms = mol.GetNumAtoms()
            
            if num_atoms > max_atoms_direct:
                # Use fragment-based approach
                logger.info(f"Using fragment-based parameterization for {mol_name} ({num_atoms} atoms)")
                params = self._generate_polymer_params_fragmented(
                    model, mol_name, smiles
                )
            else:
                # Use standard approach
                params = self._generate_small_molecule_params(
                    model, mol_name, smiles, force
                )
                
            all_nb_params[mol_name] = params
            
        return all_nb_params
    
    def _generate_polymer_params_fragmented(self, model, mol_name: str, polymer_smiles: str):
        """
        Generate parameters by fragmenting polymer into representative units.
        
        Strategy:
        1. Identify unique chemical environments (monomer types, end groups)
        2. Generate parameters for each fragment
        3. Map parameters back to full chain
        
        Args:
            model: Loaded ByteFF2 model
            mol_name: Name of the polymer
            polymer_smiles: Full polymer SMILES
            
        Returns:
            Parameters dictionary for the full polymer
        """
        from byteff2.toolkit.polymer_builder import fragment_polymer_for_params
        
        # Get fragments and their mappings
        fragments, atom_mapping = fragment_polymer_for_params(polymer_smiles)
        
        # Generate parameters for each unique fragment
        fragment_params = {}
        for frag_name, frag_smiles in fragments.items():
            fragment_params[frag_name] = self._generate_small_molecule_params(
                model, frag_name, frag_smiles, force=False
            )
            
        # Map fragment parameters back to full polymer
        full_params = self._map_fragment_params_to_polymer(
            fragment_params, atom_mapping, polymer_smiles
        )
        
        return full_params
    
    def _generate_small_molecule_params(self, model, mol_name: str, smiles: str, force: bool):
        """
        Generate parameters for a small molecule using standard ByteFF2 approach.
        
        Args:
            model: Loaded ByteFF2 model
            mol_name: Molecule name
            smiles: SMILES string
            force: Force regeneration
            
        Returns:
            Parameters dictionary
        """
        # Use existing parameter generation logic
        params_file = os.path.join(self.params_dir, f"{mol_name}_params.json")
        
        if os.path.exists(params_file) and not force:
            with open(params_file, 'r') as f:
                return json.load(f)
                
        # Generate new parameters using model
        from byteff2.toolkit.param_generator import generate_params_from_smiles
        
        params = generate_params_from_smiles(model, smiles, mol_name)
        
        # Save parameters
        os.makedirs(self.params_dir, exist_ok=True)
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
            
        return params
    
    def _map_fragment_params_to_polymer(self, fragment_params: dict, 
                                         atom_mapping: dict, 
                                         polymer_smiles: str) -> dict:
        """
        Map fragment parameters back to the full polymer.
        
        Args:
            fragment_params: Parameters for each fragment
            atom_mapping: Mapping from polymer atoms to fragment atoms
            polymer_smiles: Full polymer SMILES
            
        Returns:
            Full polymer parameters
        """
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(polymer_smiles)
        num_atoms = mol.GetNumAtoms()
        
        # Initialize full parameter arrays
        full_params = {
            'charges': [0.0] * num_atoms,
            'sigmas': [0.0] * num_atoms,
            'epsilons': [0.0] * num_atoms,
            'alphas': [0.0] * num_atoms,  # Polarizabilities
        }
        
        # Map each atom's parameters from its fragment
        for poly_idx, (frag_name, frag_idx) in atom_mapping.items():
            frag_p = fragment_params[frag_name]
            full_params['charges'][poly_idx] = frag_p['charges'][frag_idx]
            full_params['sigmas'][poly_idx] = frag_p['sigmas'][frag_idx]
            full_params['epsilons'][poly_idx] = frag_p['epsilons'][frag_idx]
            if 'alphas' in frag_p:
                full_params['alphas'][poly_idx] = frag_p['alphas'][frag_idx]
                
        return full_params


class PolymerDensityProtocol(PolymerElectrolyteProtocol, DensityProtocol):
    """
    Protocol for computing density of polymer electrolyte systems.
    
    Extends base DensityProtocol with polymer-specific box building
    and equilibration procedures.
    """
    
    def __init__(self, output_dir: str, params_dir: Optional[str] = None):
        PolymerElectrolyteProtocol.__init__(self, output_dir, params_dir)
        
    def run(self):
        """Run the polymer density protocol."""
        logger.info("Running Polymer Density Protocol")
        
        # Build polymer chains if needed
        if self.polymer_components:
            struct_dir = os.path.join(self.output_dir, "structures")
            self.build_polymer_chains(struct_dir)
            
        # Build simulation box
        box_builder = self.get_box_builder()
        system_gro = box_builder.build_box(
            components=self._get_all_components(),
            box_size=self._estimate_box_size(),
            output_dir=self.output_dir,
            target_density=self.config.get("target_density"),
        )
        
        # Generate force field parameters
        smiles_dict = {c.name: c.smiles for c in self._get_all_components()}
        self.generate_ff_params_polymer(smiles_dict)
        
        # Run equilibration and production
        self._run_polymer_equilibration()
        self._run_density_production()
        
    def _get_all_components(self) -> List[Component]:
        """Get all components including polymers."""
        components = list(self.polymer_components.values())
        # Add non-polymer components from config
        for name, comp in self.config.get("components", {}).items():
            if isinstance(comp, dict) and comp.get("type") != "polymer":
                smiles = self.config.get("smiles", {}).get(name, "")
                components.append(Component(
                    name=name,
                    smiles=smiles,
                    type=ComponentType.CATION if "cation" in comp.get("type", "") 
                         else ComponentType.ANION if "anion" in comp.get("type", "")
                         else ComponentType.SOLVENT,
                    molar_num=comp.get("count", 1),
                ))
        return components
    
    def _estimate_box_size(self) -> float:
        """Estimate initial box size for polymer system."""
        target_density = self.config.get("target_density", 1.0)
        
        # Calculate total mass
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        total_mass = 0.0
        for comp in self._get_all_components():
            mol = Chem.MolFromSmiles(comp.smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                total_mass += mw * comp.molar_num
                
        # V = m / rho, convert to nm
        # mass in g/mol, density in g/cm³
        # V in cm³ = mass / (density * N_A) where N_A ~ 6.022e23
        # For simulation, we use nm³
        import math
        N_A = 6.022e23
        volume_cm3 = total_mass / (target_density * N_A)
        volume_nm3 = volume_cm3 * 1e21  # cm³ to nm³
        box_length_nm = math.pow(volume_nm3, 1/3)
        
        # Add buffer for initial packing
        buffer_factor = self.config.get("box_buffer_factor", 1.2)
        return box_length_nm * buffer_factor
    
    def _run_polymer_equilibration(self):
        """Run multi-stage equilibration for polymer system."""
        from byteff2.toolkit.polymer_simulation import PolymerEquilibrationProtocol
        
        equil_protocol = PolymerEquilibrationProtocol(self.config)
        equil_protocol.run(
            topology_file=os.path.join(self.output_dir, "topol.top"),
            structure_file=os.path.join(self.output_dir, "system.gro"),
            output_dir=self.output_dir,
        )
        
    def _run_density_production(self):
        """Run production simulation for density calculation."""
        # Use parent class production run
        pass


class PolymerTransportProtocol(PolymerElectrolyteProtocol, TransportProtocol):
    """
    Protocol for computing transport properties of polymer electrolytes.
    
    Additional analyses beyond standard TransportProtocol:
    - Polymer chain diffusion (center of mass)
    - Ion hopping statistics
    - Coordination number dynamics
    - Transference number via concentrated solution theory
    """
    
    # def __init__(self, output_dir: str, params_dir: Optional[str] = None):
    #     PolymerElectrolyteProtocol.__init__(self, output_dir, params_dir)

    # def __init__(self, output_dir: str, params_dir: Optional[str] = None):
    #     # Initialize PolymerElectrolyteProtocol attributes directly
    #     self.output_dir = output_dir
    #     self.params_dir = params_dir or os.path.join(output_dir, "params")
    #     self.config = {}
    #     self.polymer_components: Dict[str, PolymerComponent] = {}
    #     self.box_builder_type = BoxBuilderType.PACKMOL
    #     self._polymer_builder = None
    def __init__(self, output_dir: str, params_dir: Optional[str] = None):
        # Only call PolymerElectrolyteProtocol.__init__ to avoid MRO issues
        # with TransportProtocol expecting a config dict
        resolved_params_dir = params_dir or os.path.join(output_dir, "params")
        
        # Initialize Protocol attributes directly (bypass TransportProtocol's __init__)
        os.makedirs(resolved_params_dir, exist_ok=True)
        self.params_dir = resolved_params_dir
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # PolymerElectrolyteProtocol attributes
        self.config = {}
        self.polymer_components: Dict[str, PolymerComponent] = {}
        self.box_builder_type = BoxBuilderType.PACKMOL
        self._polymer_builder = None
        
        # TransportProtocol attributes
        self.components = None

    def run(self):
        """Run the polymer transport protocol."""
        logger.info("Running Polymer Transport Protocol")
        
        # Build polymer chains if needed
        if self.polymer_components:
            struct_dir = os.path.join(self.output_dir, "structures")
            self.build_polymer_chains(struct_dir)
            
        # Build simulation box using Packmol
        box_builder = self.get_box_builder()
        system_gro = box_builder.build_box(
            components=self._get_all_components(),
            box_size=self._estimate_box_size(),
            output_dir=self.output_dir,
            target_density=self.config.get("target_density"),
        )
        
        # Generate force field parameters
        smiles_dict = {c.name: c.smiles for c in self._get_all_components()}
        self.generate_ff_params_polymer(smiles_dict)
        
        # Run equilibration
        self._run_polymer_equilibration()
        
        # Run production NVT for transport properties
        self._run_transport_production()
        
        # Post-process with polymer-specific analyses
        self.post_process_polymer()
        
    def _get_all_components(self) -> List[Component]:
        """Get all components including polymers."""
        components = list(self.polymer_components.values())
        for name, comp in self.config.get("components", {}).items():
            if isinstance(comp, dict) and comp.get("type") != "polymer":
                smiles = self.config.get("smiles", {}).get(name, "")
                components.append(Component(
                    name=name,
                    smiles=smiles,
                    type=ComponentType.CATION if "cation" in comp.get("type", "") 
                         else ComponentType.ANION if "anion" in comp.get("type", "")
                         else ComponentType.SOLVENT,
                    molar_num=comp.get("count", 1),
                ))
        return components
    
    def _estimate_box_size(self) -> float:
        """Estimate initial box size for polymer system."""
        target_density = self.config.get("target_density", 1.0)
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        import math
        
        total_mass = 0.0
        for comp in self._get_all_components():
            mol = Chem.MolFromSmiles(comp.smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                total_mass += mw * comp.molar_num
                
        N_A = 6.022e23
        volume_cm3 = total_mass / (target_density * N_A)
        volume_nm3 = volume_cm3 * 1e21
        box_length_nm = math.pow(volume_nm3, 1/3)
        
        buffer_factor = self.config.get("box_buffer_factor", 1.2)
        return box_length_nm * buffer_factor
    
    def _run_polymer_equilibration(self):
        """Run multi-stage equilibration for polymer system."""
        from byteff2.toolkit.polymer_simulation import PolymerEquilibrationProtocol
        
        equil_protocol = PolymerEquilibrationProtocol(self.config)
        equil_protocol.run(
            topology_file=os.path.join(self.output_dir, "topol.top"),
            structure_file=os.path.join(self.output_dir, "system.gro"),
            output_dir=self.output_dir,
        )
        
    def _run_transport_production(self):
        """Run NVT production for transport property calculation."""
        pass
        
    def post_process_polymer(self):
        """Extended post-processing for polymer systems."""
        results = {}
        
        # Standard transport analysis
        if self.config.get("compute_conductivity", True):
            results["conductivity"] = self._compute_ionic_conductivity()
            
        if self.config.get("compute_viscosity", True):
            results["viscosity"] = self._compute_viscosity()
            
        # Polymer-specific analyses
        if self.config.get("compute_chain_diffusion", True):
            results["chain_diffusion"] = self._compute_chain_diffusion()
            
        if self.config.get("compute_ion_coordination", True):
            results["ion_coordination"] = self._compute_ion_coordination()
            
        if self.config.get("compute_ion_hopping", False):
            results["ion_hopping"] = self._compute_ion_hopping_statistics()
            
        # Save results
        results_file = os.path.join(self.output_dir, "polymer_transport_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Polymer transport results saved to {results_file}")
        return results
    
    def _compute_ionic_conductivity(self):
        """Compute ionic conductivity from trajectory."""
        # Placeholder - would use existing transport analysis
        return None
    
    def _compute_viscosity(self):
        """Compute viscosity from non-equilibrium MD."""
        # Placeholder - would use existing viscosity analysis
        return None
    
    def _compute_chain_diffusion(self):
        """
        Compute polymer chain center-of-mass diffusion.
        
        Returns:
            Dictionary with chain diffusion coefficients and MSD data
        """
        logger.info("Computing polymer chain diffusion")
        # Placeholder for chain COM MSD analysis
        return {"D_chain": None, "msd_data": None}
    
    def _compute_ion_coordination(self):
        """
        Analyze ion-polymer coordination over trajectory.
        
        Returns:
            Dictionary with coordination number statistics
        """
        logger.info("Computing ion-polymer coordination")
        # Placeholder for coordination analysis
        return {"avg_coordination": None, "coordination_histogram": None}
    
    def _compute_ion_hopping_statistics(self):
        """
        Compute ion hopping statistics between coordination sites.
        
        Returns:
            Dictionary with hopping rates and residence times
        """
        logger.info("Computing ion hopping statistics")
        # Placeholder for hopping analysis
        return {"hopping_rate": None, "residence_time": None}


def get_polymer_protocol(config: dict) -> Protocol:
    """
    Factory function to get appropriate polymer protocol.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Appropriate Protocol instance
    """
    protocol_name = config.get("protocol", "")
    params_dir = config.get("params_dir", "./params")
    output_dir = config.get("output_dir", "./output")
    
    protocol_map = {
        "PolymerDensity": PolymerDensityProtocol,
        "PolymerTransport": PolymerTransportProtocol,
        "Density": PolymerDensityProtocol,  # Auto-detect polymer
        "Transport": PolymerTransportProtocol,
    }
    
    protocol_class = protocol_map.get(protocol_name)
    if protocol_class is None:
        raise ValueError(f"Unknown polymer protocol: {protocol_name}")
        
    protocol = protocol_class(output_dir, params_dir)
    protocol.setup_from_config(config)
    return protocol