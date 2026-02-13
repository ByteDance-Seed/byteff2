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
    write_gro,
)

## added on 02-04-2026
import openmm.app as app
import openmm.unit as ou
from openmm import Vec3
from byteff2.toolkit.openmmtool import generate_openmm_system
from byteff2.md_utils.md_run import npt_run, nvt_run, rescale_box, dcd_read, volume_calc
from byteff2.md_utils.viscosity import nonequ_run, viscosity_calc
from byteff2.md_utils.onsager_conductivity import onsager_calc
###

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
class SimpleComponent:
    """Simple component class for polymer protocol internal use."""
    name: str
    smiles: str
    type: ComponentType
    molar_num: int = 1

@dataclass
class SimplePolymerComponent:
    """Simple polymer component class that doesn't require topo_mol."""
    name: str
    smiles: str
    type: ComponentType
    molar_num: int = 1
    monomer_smiles: str = ""
    degree_of_polymerization: int = 1
    end_groups: Tuple[Optional[str], Optional[str]] = (None, None)
    tacticity: str = "atactic"

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
            # polymer_type=PolymerType(config.get("polymer_type", "linear")),
            # degree_of_polymerization=config.get("dp", config.get("degree_of_polymerization", 1)),
            # monomer_smiles=config.get("monomer_smiles"),
            # end_groups=(config.get("left_cap"), config.get("right_cap")),
            # tacticity=config.get("tacticity", "atactic"),
        )

class PolymerElectrolyteProtocol(Protocol):
    """
    Base protocol for polymer electrolyte MD simulations.
    
    This protocol follows the same workflow as liquid electrolyte protocols
    (NPT → NVT → nonequ) while supporting polymer-specific features like
    polymer chain building and Packmol-based box construction.
    """
    
    def __init__(self, output_dir: str, params_dir: Optional[str] = None):
        # Base Protocol expects (params_dir, output_dir) in that order
        resolved_params_dir = params_dir or os.path.join(output_dir, "params")
        super().__init__(resolved_params_dir, output_dir)
        self.config = {}
        # self.polymer_components: Dict[str, PolymerComponent] = {}
        self.polymer_components: Dict[str, SimplePolymerComponent] = {}  # Changed type
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
            if isinstance(comp_config, dict) and comp_config.get("type") == "polymer":
                # Extract end groups properly
                end_groups_config = comp_config.get("end_groups", {})
                if isinstance(end_groups_config, dict):
                    left_cap = end_groups_config.get("left")
                    right_cap = end_groups_config.get("right")
                elif isinstance(end_groups_config, (list, tuple)) and len(end_groups_config) >= 2:
                    left_cap = end_groups_config[0]
                    right_cap = end_groups_config[1]
                else:
                    left_cap = None
                    right_cap = None

                self.polymer_components[name] = SimplePolymerComponent(
                    name=name,
                    smiles=smiles_dict.get(name, ""),
                    type=ComponentType.SOLVENT,
                    molar_num=comp_config.get("count", 1),
                    monomer_smiles=comp_config.get("monomer_smiles", ""),
                    degree_of_polymerization=comp_config.get("degree_of_polymerization", 1),
                    end_groups=(left_cap, right_cap),
                    tacticity=comp_config.get("tacticity", "atactic"),
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
                
                # # Update SMILES to full polymer SMILES
                # polymer.smiles = builder.get_polymer_smiles()
                # NOTE: Do NOT overwrite polymer.smiles with the full chain SMILES.
                # polymer.smiles is used for parameterization and must remain a small
                # representative oligomer (set by _get_all_components via trimer builder).
                # Store the full chain SMILES separately if needed.
                polymer._full_chain_smiles = builder.get_polymer_smiles()
                
                logger.info(f"Built polymer chain {name} with DP={polymer.degree_of_polymerization}")
                
        return generated_structures
    
    def generate_ff_params_polymer(self, component_smiles: dict, force: bool = False):
        """
        Generate force field parameters for polymer systems.
        
        For polymers, we use a representative oligomer (trimer) for ML parameterization,
        but the actual topology (.itp) files must match the FULL polymer chain that
        will be used in the simulation.
        
        Args:
            component_smiles: Dictionary of component names to SMILES strings
                             (for polymers, this should be the REPRESENTATIVE oligomer SMILES)
            force: Force regeneration even if parameters exist
            
        Returns:
            Dictionary of non-bonded parameters for each component
        """
        from rdkit import Chem
        from byteff2.train.utils import get_nb_params, load_model
        from bytemol.utils import get_data_file_path
        
        model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
        model = load_model(os.path.dirname(model_dir))
        all_nb_params = {}
        
        max_atoms_direct = self.config.get("max_atoms_direct_param", 500)
        
        for mol_name, smiles in component_smiles.items():
            logger.info(f'Preparing force field params for {mol_name}')
            
            # Check if this is a polymer component
            is_polymer = mol_name in self.polymer_components
            
            if is_polymer:
                polymer_comp = self.polymer_components[mol_name]
                full_dp = polymer_comp.degree_of_polymerization
                
                # For polymers, we need to:
                # 1. Use oligomer SMILES for ML parameter prediction (charges, LJ params)
                # 2. Generate full-chain topology (.itp) with proper atom count
                
                # The input smiles is already the representative oligomer
                oligomer_smiles = smiles
                
                mol = Chem.MolFromSmiles(oligomer_smiles)
                if mol is None:
                    logger.warning(f"Could not parse SMILES for {mol_name}: {oligomer_smiles}")
                    continue
                
                # Generate parameters using the oligomer
                params = self._generate_small_molecule_params(
                    model, mol_name, oligomer_smiles, force
                )
                
                # Now generate FULL polymer topology if DP > oligomer DP
                oligomer_dp = min(3, full_dp)
                if full_dp > oligomer_dp:
                    logger.info(f"Generating full polymer topology for {mol_name} (DP={full_dp})")
                    self._generate_full_polymer_topology(
                        mol_name, polymer_comp, params, force
                    )
            else:
                # Non-polymer: standard parameterization
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Could not parse SMILES for {mol_name}: {smiles}")
                    continue
                    
                num_atoms = mol.GetNumAtoms()
                
                if num_atoms > max_atoms_direct:
                    logger.info(f"Using fragment-based parameterization for {mol_name} ({num_atoms} atoms)")
                    params = self._generate_polymer_params_fragmented(model, mol_name, smiles)
                else:
                    params = self._generate_small_molecule_params(model, mol_name, smiles, force)
                    
            all_nb_params[mol_name] = params
        
        # Load metadata from any component's nb_params file
        for mol_name in component_smiles.keys():
            nb_meta_fp = os.path.join(self.params_dir, f'{mol_name}_nb_params.json')
            if os.path.isfile(nb_meta_fp):
                try:
                    with open(nb_meta_fp, 'r') as f:
                        meta_wrap = json.load(f)
                        if isinstance(meta_wrap, dict) and 'metadata' in meta_wrap:
                            all_nb_params['metadata'] = meta_wrap['metadata']
                            break
                except Exception:
                    pass
            
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

        from byteff2.train.utils import get_nb_params
        from byteff2.toolkit.protocol import write_gro
        from bytemol.core import Molecule
        
        # Check for cached parameters
        itp_fp = os.path.join(self.params_dir, f'{mol_name}.itp')
        atp_fp = os.path.join(self.params_dir, f'{mol_name}.atp')
        gro_fp = os.path.join(self.params_dir, f'{mol_name}.gro')
        params_json_fp = os.path.join(self.params_dir, f'{mol_name}.json')
        nb_meta_fp = os.path.join(self.params_dir, f'{mol_name}_nb_params.json')
        
        have_all = all(os.path.isfile(p) for p in (itp_fp, atp_fp, gro_fp, params_json_fp))
        
        if have_all and not force:
            logger.info(f'Found cached params for {mol_name}; loading from {params_json_fp}')
            with open(params_json_fp, 'r') as f:
                return json.load(f)
        
        # Generate new parameters using model
        logger.info(f'Generating force field params for {mol_name}')
        mol = Molecule.from_smiles(smiles, nconfs=1)
        mol.name = mol_name
        
        metadata, params, tfs, mol = get_nb_params(model, mol)
        
        # Save ITP and ATP files
        os.makedirs(self.params_dir, exist_ok=True)
        tfs.write_itp(itp_fp, separated_atp=True)
        
        # Save GRO file
        write_gro(mol, gro_fp)
        
        # Save parameters JSON
        with open(params_json_fp, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Save metadata
        with open(nb_meta_fp, 'w') as f:
            json.dump({'metadata': metadata}, f, indent=2)
            
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

    def _get_all_components(self) -> List[SimpleComponent]:
        """Get all components including polymers."""
        components: List[SimpleComponent] = []
        
        # First add polymer components
        for name, polymer_comp in self.polymer_components.items():
            smiles = polymer_comp.smiles
            
            # Always generate a small representative oligomer for parameterization
            # (not the full chain, which can be hundreds of atoms)
            if (not smiles or smiles == "") and polymer_comp.monomer_smiles:
                try:
                    from byteff2.toolkit.polymer_builder import PolymerChainBuilder
                    
                    left_cap = polymer_comp.end_groups[0] if polymer_comp.end_groups else None
                    right_cap = polymer_comp.end_groups[1] if polymer_comp.end_groups else None
                    
                    # Use a small representative oligomer (trimer) for parameterization
                    # This ensures the SMILES is small enough for direct parameterization
                    # and avoids the fragmentation path entirely
                    param_dp = min(3, polymer_comp.degree_of_polymerization)
                    
                    builder = PolymerChainBuilder(
                        monomer_smiles=polymer_comp.monomer_smiles,
                        dp=param_dp,
                        end_group_left=left_cap,
                        end_group_right=right_cap,
                        tacticity=polymer_comp.tacticity,
                    )
                    smiles = builder.get_polymer_smiles()
                    polymer_comp.smiles = smiles
                    logger.info(f"Generated representative oligomer SMILES for {name} (DP={param_dp}): {smiles}")
                except Exception as e:
                    logger.warning(f"Could not generate polymer SMILES for {name}: {e}")
                    smiles = polymer_comp.monomer_smiles.replace('[*]', '')
            
            components.append(SimpleComponent(
                name=name,
                smiles=smiles,
                type=polymer_comp.type,
                molar_num=polymer_comp.molar_num,
            ))
        
        # Add non-polymer components from config
        for name, comp in self.config.get("components", {}).items():
            if isinstance(comp, dict) and comp.get("type") != "polymer":
                smiles = self.config.get("smiles", {}).get(name, "")
                comp_type_str = comp.get("type", "")
                if "cation" in comp_type_str:
                    comp_type = ComponentType.CATION
                elif "anion" in comp_type_str:
                    comp_type = ComponentType.ANION
                else:
                    comp_type = ComponentType.SOLVENT
                    
                components.append(SimpleComponent(
                    name=name,
                    smiles=smiles,
                    type=comp_type,
                    molar_num=comp.get("count", 1),
                ))
        
        logger.info(f"Total components: {[c.name for c in components]}")
        return components
    
    def _generate_full_polymer_topology(self, mol_name: str, polymer_comp, 
                                         oligomer_params: dict, force: bool = False):
        """
        Generate full polymer topology (.itp, .atp, .gro) files.
        
        This creates topology files for the FULL polymer chain (e.g., DP=130),
        using parameters derived from the oligomer but with the correct atom count.
        
        Args:
            mol_name: Polymer name
            polymer_comp: SimplePolymerComponent with monomer info
            oligomer_params: Parameters from oligomer parameterization
            force: Force regeneration
        """
        from byteff2.toolkit.polymer_builder import PolymerChainBuilder
        from byteff2.toolkit.protocol import write_gro
        from bytemol.core import Molecule
        
        itp_fp = os.path.join(self.params_dir, f'{mol_name}.itp')
        atp_fp = os.path.join(self.params_dir, f'{mol_name}.atp')
        gro_fp = os.path.join(self.params_dir, f'{mol_name}.gro')
        
        # Check if we need to regenerate
        if not force and all(os.path.isfile(p) for p in [itp_fp, atp_fp, gro_fp]):
            # Verify the existing files have the correct atom count
            try:
                with open(gro_fp, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        existing_natoms = int(lines[1].strip())
                        # Build expected atom count
                        builder = PolymerChainBuilder(
                            monomer_smiles=polymer_comp.monomer_smiles,
                            dp=polymer_comp.degree_of_polymerization,
                            end_group_left=polymer_comp.end_groups[0],
                            end_group_right=polymer_comp.end_groups[1],
                            tacticity=polymer_comp.tacticity,
                        )
                        # Don't fully build, just estimate atom count
                        from rdkit import Chem
                        monomer = Chem.MolFromSmiles(polymer_comp.monomer_smiles)
                        monomer = Chem.AddHs(monomer)
                        # Each monomer contributes its atoms minus 2 dummies
                        # Plus end groups
                        atoms_per_monomer = monomer.GetNumAtoms() - 2  # Remove dummy atoms
                        expected_natoms = atoms_per_monomer * polymer_comp.degree_of_polymerization
                        # Add end group atoms (approximate)
                        expected_natoms += 4  # Rough estimate for end groups
                        
                        if abs(existing_natoms - expected_natoms) < expected_natoms * 0.1:
                            logger.info(f"Found valid full polymer topology for {mol_name} ({existing_natoms} atoms)")
                            return
            except Exception:
                pass
        
        logger.info(f"Building full polymer chain for topology: {mol_name}, DP={polymer_comp.degree_of_polymerization}")
        
        # Build the full polymer chain
        builder = PolymerChainBuilder(
            monomer_smiles=polymer_comp.monomer_smiles,
            dp=polymer_comp.degree_of_polymerization,
            end_group_left=polymer_comp.end_groups[0],
            end_group_right=polymer_comp.end_groups[1],
            tacticity=polymer_comp.tacticity,
        )
        
        full_mol = builder.build_chain()
        full_smiles = builder.get_polymer_smiles()
        
        logger.info(f"Full polymer has {full_mol.GetNumAtoms()} atoms")
        
        # Generate parameters for the full molecule
        # We can't use ML for 652 atoms, so we extrapolate from oligomer
        full_params = self._extrapolate_params_to_full_chain(
            oligomer_params, polymer_comp, full_mol
        )
        
        # Create Molecule object and generate topology files
        try:
            mol_obj = Molecule.from_smiles(full_smiles, nconfs=1)
            mol_obj.name = mol_name
        except Exception as e:
            logger.warning(f"Could not create Molecule from full SMILES: {e}")
            # Fallback: use RDKit mol directly
            mol_obj = self._create_molecule_from_rdkit(full_mol, mol_name)
        
        # Generate topology using byteff2's standard approach but with extrapolated params
        from byteff2.train.utils import get_nb_params, load_model
        from bytemol.utils import get_data_file_path
        
        # We need to generate proper ITP/ATP files for the full polymer
        # This requires using the TopoForceSystem from bytemol
        self._write_polymer_topology_files(
            mol_name, full_mol, full_params, itp_fp, atp_fp
        )
        
        # Write GRO file
        write_gro(mol_obj, gro_fp)
        
        logger.info(f"Generated full polymer topology files for {mol_name}")

    def _extrapolate_params_to_full_chain(self, oligomer_params: dict, 
                                           polymer_comp, full_mol) -> dict:
        """
        Extrapolate parameters from oligomer to full polymer chain.
        
        Strategy:
        - Interior repeat units use the same parameters as the middle unit of the oligomer
        - End groups use parameters from the oligomer's end groups
        
        Args:
            oligomer_params: Parameters from trimer
            polymer_comp: Polymer component info
            full_mol: Full polymer RDKit molecule
            
        Returns:
            Parameters for full polymer
        """
        from rdkit import Chem
        from byteff2.toolkit.polymer_builder import PolymerChainBuilder
        
        num_atoms = full_mol.GetNumAtoms()
        
        # Get oligomer size for reference
        oligomer_dp = min(3, polymer_comp.degree_of_polymerization)
        
        # Build a reference oligomer to understand the parameter pattern
        ref_builder = PolymerChainBuilder(
            monomer_smiles=polymer_comp.monomer_smiles,
            dp=oligomer_dp,
            end_group_left=polymer_comp.end_groups[0],
            end_group_right=polymer_comp.end_groups[1],
            tacticity=polymer_comp.tacticity,
        )
        ref_mol = ref_builder.build_chain()
        ref_natoms = ref_mol.GetNumAtoms()
        
        # Parse monomer to understand repeat unit structure
        monomer = Chem.MolFromSmiles(polymer_comp.monomer_smiles)
        monomer = Chem.AddHs(monomer)
        # Atoms per repeat unit (excluding dummy atoms)
        atoms_in_monomer = monomer.GetNumAtoms()
        dummy_count = sum(1 for a in monomer.GetAtoms() if a.GetAtomicNum() == 0)
        atoms_per_repeat = atoms_in_monomer - dummy_count
        
        # Initialize full parameter arrays
        full_params = {
            'charge': [0.0] * num_atoms,
            'sigma': [0.0] * num_atoms,
            'epsilon': [0.0] * num_atoms,
        }
        
        # Copy alpha if present
        if 'alpha' in oligomer_params:
            full_params['alpha'] = [0.0] * num_atoms
        
        # Get oligomer parameters - handle both possible key names
        oligomer_charges = oligomer_params.get('charge', oligomer_params.get('charges', []))
        oligomer_sigmas = oligomer_params.get('sigma', oligomer_params.get('sigmas', []))
        oligomer_epsilons = oligomer_params.get('epsilon', oligomer_params.get('epsilons', []))
        oligomer_alphas = oligomer_params.get('alpha', oligomer_params.get('alphas', []))
        
        if not oligomer_charges or len(oligomer_charges) != ref_natoms:
            logger.warning(f"Oligomer param mismatch: expected {ref_natoms}, got {len(oligomer_charges)}")
            logger.warning("Using uniform fallback parameters")
            # Fallback: use uniform parameters
            avg_charge = 0.0
            avg_sigma = 0.35  # nm
            avg_epsilon = 0.3  # kJ/mol
            avg_alpha = 0.001  # nm³
            
            for i in range(num_atoms):
                full_params['charge'][i] = avg_charge
                full_params['sigma'][i] = avg_sigma
                full_params['epsilon'][i] = avg_epsilon
                if 'alpha' in full_params:
                    full_params['alpha'][i] = avg_alpha
            return full_params
        
        # Map oligomer parameters to full chain using periodicity
        # The middle repeat unit (indices roughly in center of oligomer) is representative
        middle_start = atoms_per_repeat  # Skip first repeat unit (includes end effects)
        
        for i in range(num_atoms):
            if i < atoms_per_repeat:
                # Left end region
                ref_idx = min(i, ref_natoms - 1)
            elif i >= num_atoms - atoms_per_repeat:
                # Right end region
                offset_from_end = num_atoms - 1 - i
                ref_idx = ref_natoms - 1 - min(offset_from_end, atoms_per_repeat - 1)
            else:
                # Interior: periodic mapping
                interior_idx = (i - atoms_per_repeat) % atoms_per_repeat
                ref_idx = min(middle_start + interior_idx, ref_natoms - 1)
            
            ref_idx = max(0, min(ref_idx, len(oligomer_charges) - 1))
            
            full_params['charge'][i] = oligomer_charges[ref_idx]
            full_params['sigma'][i] = oligomer_sigmas[ref_idx] if oligomer_sigmas else 0.35
            full_params['epsilon'][i] = oligomer_epsilons[ref_idx] if oligomer_epsilons else 0.3
            if 'alpha' in full_params and oligomer_alphas:
                full_params['alpha'][i] = oligomer_alphas[ref_idx] if ref_idx < len(oligomer_alphas) else 0.001
        
        # Ensure charge neutrality
        total_charge = sum(full_params['charge'])
        if abs(total_charge) > 0.01:
            correction = -total_charge / num_atoms
            full_params['charge'] = [c + correction for c in full_params['charge']]
            logger.info(f"Adjusted polymer charges for neutrality (correction: {correction:.6f} per atom)")
        
        return full_params

    def _write_polymer_topology_files(self, mol_name: str, mol, params: dict,
                                       itp_path: str, atp_path: str):
        """
        Write ITP and ATP topology files for a polymer.
        
        This is a simplified topology writer that creates GROMACS-compatible
        files for the full polymer chain.
        """
        from rdkit import Chem
        
        num_atoms = mol.GetNumAtoms()
        
        # Generate atom types based on element and index
        atom_types = []
        for i, atom in enumerate(mol.GetAtoms()):
            elem = atom.GetSymbol()
            atom_types.append(f"{mol_name}_{elem}{i+1}")
        
        # Write ATP file (atom types)
        with open(atp_path, 'w') as f:
            f.write("; Atom types for {}\n".format(mol_name))
            f.write("[ atomtypes ]\n")
            f.write("; name    at.num    mass    charge    ptype    sigma    epsilon\n")
            
            for i, atom in enumerate(mol.GetAtoms()):
                atype = atom_types[i]
                at_num = atom.GetAtomicNum()
                mass = atom.GetMass()
                charge = params['charge'][i] if i < len(params.get('charge', [])) else 0.0
                sigma = params['sigma'][i] if i < len(params.get('sigma', [])) else 0.35
                epsilon = params['epsilon'][i] if i < len(params.get('epsilon', [])) else 0.3
                
                f.write(f"  {atype:10s}  {at_num:3d}  {mass:8.4f}  {charge:8.5f}  A  {sigma:10.6f}  {epsilon:10.6f}\n")
        
        # Write ITP file (molecule topology)
        with open(itp_path, 'w') as f:
            f.write("; Topology for {}\n".format(mol_name))
            f.write("[ moleculetype ]\n")
            f.write("; name    nrexcl\n")
            f.write(f"  {mol_name}    3\n\n")
            
            f.write("[ atoms ]\n")
            f.write("; nr    type    resnr    residu    atom    cgnr    charge    mass\n")
            
            for i, atom in enumerate(mol.GetAtoms()):
                atype = atom_types[i]
                mass = atom.GetMass()
                charge = params['charge'][i] if i < len(params.get('charge', [])) else 0.0
                elem = atom.GetSymbol()
                # FIX: atom name should be a string, not mixed format
                atom_name = f"{elem}{i+1}"
                
                f.write(f"  {i+1:5d}  {atype:10s}  1  {mol_name[:3]:4s}  {atom_name:>5s}  {i+1:5d}  {charge:10.6f}  {mass:10.4f}\n")
            
            # Write bonds
            f.write("\n[ bonds ]\n")
            f.write("; ai    aj    funct    c0    c1\n")
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx() + 1
                j = bond.GetEndAtomIdx() + 1
                f.write(f"  {i:5d}  {j:5d}  1\n")
            
            # Write angles (basic angle detection)
            f.write("\n[ angles ]\n")
            f.write("; ai    aj    ak    funct    c0    c1\n")
            
            for atom in mol.GetAtoms():
                j = atom.GetIdx()
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(neighbors) >= 2:
                    for idx1 in range(len(neighbors)):
                        for idx2 in range(idx1 + 1, len(neighbors)):
                            i = neighbors[idx1] + 1
                            k = neighbors[idx2] + 1
                            f.write(f"  {i:5d}  {j+1:5d}  {k:5d}  1\n")
            
            # Write dihedrals (basic detection)
            f.write("\n[ dihedrals ]\n")
            f.write("; ai    aj    ak    al    funct    c0    c1    c2    c3\n")
            
            for bond in mol.GetBonds():
                j = bond.GetBeginAtomIdx()
                k = bond.GetEndAtomIdx()
                j_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
                k_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
                
                for i in j_neighbors:
                    for l in k_neighbors:
                        f.write(f"  {i+1:5d}  {j+1:5d}  {k+1:5d}  {l+1:5d}  3\n")
        
        logger.info(f"Written polymer topology: {itp_path}, {atp_path}")

    def _create_molecule_from_rdkit(self, rdkit_mol, name: str):
        """Create a bytemol Molecule from an RDKit mol."""
        from bytemol.core import Molecule
        import tempfile
        
        # Write to temporary file and read back
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
            temp_path = f.name
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Ensure 3D coordinates
            if rdkit_mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDGv3())
            
            Chem.MolToPDBFile(rdkit_mol, temp_path)
            mol = Molecule.from_file(temp_path)
            mol.name = name
            return mol
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

class PolymerDensityProtocol(PolymerElectrolyteProtocol, DensityProtocol):
    """
    Protocol for computing density of polymer electrolyte systems.
    
    Extends base DensityProtocol with polymer-specific box building
    and equilibration procedures. Uses the same NPT workflow as liquid electrolyte DensityProtocol. 
    """
    
    def __init__(self, output_dir: str, params_dir: Optional[str] = None):
        PolymerElectrolyteProtocol.__init__(self, output_dir, params_dir)

    def run_protocol(self):
        """
        Run the polymer transport protocol using the SAME workflow as liquid electrolytes.
        
        Workflow: Build system → NPT → rescale → NVT → (optional) nonequ
        This mirrors TransportProtocol.run_protocol() exactly.
        """
        logger.info("Running Polymer Density Protocol")

        # Get representative oligomer SMILES for parameterization FIRST
        smiles_dict = {}
        for comp in self._get_all_components():
            smiles_dict[comp.name] = comp.smiles
        
        # Build polymer chains if needed
        if self.polymer_components:
            struct_dir = os.path.join(self.output_dir, "structures")
            self.build_polymer_chains(struct_dir)
        
        # Generate force field parameters first
        nonbonded_params = self.generate_ff_params_polymer(smiles_dict, force=bool(self.config.get('force_regenerate_params', False)))
        
        # # Save nonbonded params for equilibration
        # params_json_path = os.path.join(self.output_dir, 'nonbonded_params.json')
        # with open(params_json_path, 'w') as f:
        #     json.dump(nonbonded_params, f, indent=2)
        
        # Build components ratio dict
        components_ratio = {}
        for comp in self._get_all_components():
            components_ratio[comp.name] = comp.molar_num
        
        # Use a separate working directory for build_system to avoid SameFileError
        # (build_system copies files from params_dir -> working_dir; they must differ)
        build_working_dir = self.config.get('working_dir', os.path.join(self.output_dir, 'build'))

        # Use base Protocol's build_system
        natoms = self.config.get("natoms", 5000)
        self.components = self.build_system(
            total_atoms=natoms,
            components_ratio=components_ratio,
            # working_dir=self.params_dir,
            working_dir=build_working_dir,
            reuse_if_exists=bool(self.config.get('resume', False)),
        )
        
        ## ensuring same workflow as liquid electrolyte density protocol
        # Load system for OpenMM (files are in params_dir)
        gro_file = os.path.join(self.params_dir, "solvent_salt.gro")
        top_file = os.path.join(self.params_dir, "system.top")
        
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        
        input_top, input_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )
        
        # Run NPT (same as liquid electrolytes)
        npt_steps = int(self.config.get('npt_steps', 1500000))
        npt_timestep_fs = int(self.config.get('npt_timestep_fs', 2))
        resume = bool(self.config.get('resume', False))
        checkpoint_interval = int(self.config.get('checkpoint_interval', 5000))
        
        npt_run(
            top=input_top,
            system=input_system,
            positions=input_positions,
            temperature=self.config['temperature'],
            npt_steps=npt_steps,
            work_dir=self.output_dir,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            timestep=npt_timestep_fs,
            state_csv_override=self.config.get('npt_state_csv'),
            dcd_path_override=self.config.get('npt_dcd'),
            resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)),
            resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)),
        )
        
        logger.info('Finished running polymer density protocol')

    # Keep old run() as alias
    def run(self):
        """Alias for run_protocol() for backward compatibility."""
        self.run_protocol()

    def post_process(self):
        """Post-process density results (same as liquid electrolytes)."""
        import pandas as pd
        
        csv_file = os.path.join(self.output_dir, 'npt_state.csv')
        density = pd.read_csv(csv_file)["Density (g/mL)"]
        
        # Use last 10% of data for averaging
        n_samples = max(1, len(density) // 10)
        density_mean = density.iloc[-n_samples:].mean()
        density_std = density.iloc[-n_samples:].std()
        
        result = {
            "density": float(density_mean),
            "density_std": float(density_std),
        }
        
        with open(os.path.join(self.output_dir, 'density_results.json'), 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Density: {result['density']:.4f} ± {result['density_std']:.4f} g/mL")
        return result

    # def _get_all_components(self) -> List[SimpleComponent]:
    #     """Get all components including polymers."""
    #     components: List[SimpleComponent] = []
        
    #     # First add polymer components
    #     for name, polymer_comp in self.polymer_components.items():
    #         # For polymers, we need to generate the full SMILES from monomer
    #         smiles = polymer_comp.smiles
            
    #         # Check if we need to generate SMILES from monomer
    #         if (not smiles or smiles == "") and polymer_comp.monomer_smiles:
    #             # Build a representative SMILES for the polymer
    #             # For parameterization, we use a trimer (3 repeat units) as representative
    #             try:
    #                 from byteff2.toolkit.polymer_builder import PolymerChainBuilder
                    
    #                 # Determine end groups
    #                 left_cap = polymer_comp.end_groups[0] if polymer_comp.end_groups else None
    #                 right_cap = polymer_comp.end_groups[1] if polymer_comp.end_groups else None
                    
    #                 # Use smaller DP for parameterization (full chain would be too large)
    #                 param_dp = min(3, polymer_comp.degree_of_polymerization)
                    
    #                 builder = PolymerChainBuilder(
    #                     monomer_smiles=polymer_comp.monomer_smiles,
    #                     dp=param_dp,
    #                     end_group_left=left_cap,
    #                     end_group_right=right_cap,
    #                     tacticity=polymer_comp.tacticity,
    #                 )
    #                 smiles = builder.get_polymer_smiles()
    #                 polymer_comp.smiles = smiles
    #                 logger.info(f"Generated polymer SMILES for {name}: {smiles}")
    #             except Exception as e:
    #                 logger.warning(f"Could not generate polymer SMILES for {name}: {e}")
    #                 # Fallback: use monomer SMILES directly (remove connection points)
    #                 smiles = polymer_comp.monomer_smiles.replace('[*]', '')
    #                 # If monomer is simple like "CCO", use it directly
    #                 if not smiles:
    #                     smiles = polymer_comp.monomer_smiles
    #                 polymer_comp.smiles = smiles
    #                 logger.info(f"Using fallback SMILES for {name}: {smiles}")
            
    #         if smiles:  # Only add if we have a valid SMILES
    #             components.append(SimpleComponent(
    #                 name=name,
    #                 smiles=smiles,
    #                 type=polymer_comp.type if hasattr(polymer_comp, 'type') else ComponentType.SOLVENT,
    #                 molar_num=polymer_comp.molar_num if hasattr(polymer_comp, 'molar_num') else 1,
    #             ))
    #         else:
    #             logger.warning(f"Skipping polymer component {name}: no SMILES available")
        
    #     # Add non-polymer components from config
    #     for name, comp in self.config.get("components", {}).items():
    #         if isinstance(comp, dict) and comp.get("type") != "polymer":
    #             smiles = self.config.get("smiles", {}).get(name, "")
    #             if smiles:  # Only add if SMILES is available
    #                 comp_type = comp.get("type", "")
    #                 if "cation" in comp_type:
    #                     component_type = ComponentType.CATION
    #                 elif "anion" in comp_type:
    #                     component_type = ComponentType.ANION
    #                 else:
    #                     component_type = ComponentType.SOLVENT
                    
    #                 components.append(SimpleComponent(
    #                     name=name,
    #                     smiles=smiles,
    #                     type=component_type,
    #                     molar_num=comp.get("count", 1),
    #                 ))
        
    #     logger.info(f"Total components: {[c.name for c in components]}")
    #     return components
    
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
        
    # def _run_polymer_equilibration(self):
    #     """Run multi-stage equilibration for polymer system."""
    #     from byteff2.toolkit.polymer_simulation import PolymerEquilibrationProtocol
        
    #     equil_protocol = PolymerEquilibrationProtocol(
    #         temperature=self.config.get("temperature", 300),
    #         pressure=self.config.get("pressure", 1.0),
    #         target_density=self.config.get("target_density"),
    #         stages=self.config.get("equilibration_stages"),
    #         output_dir=self.output_dir,
    #     )
    #     equil_protocol.run_equilibration(
    #         topology_path=os.path.join(self.output_dir, "topol.top"),
    #         coordinates_path=os.path.join(self.output_dir, "system.gro"),
    #         mdp_template_dir=self.output_dir,
    #     )

    # def _run_density_production(self):
    #     """Run production simulation for density calculation."""
    #     # Use parent class production run
    #     pass

class PolymerTransportProtocol(PolymerElectrolyteProtocol, TransportProtocol):
    """
    Protocol for computing transport properties of polymer electrolytes.

    Uses the same NPT → NVT → nonequ workflow as liquid electrolyte TransportProtocol,
    with optional polymer-specific post-processing analyses:
    - Polymer chain diffusion (center of mass)
    - Ion hopping statistics
    - Coordination number dynamics
    - Transference number via concentrated solution theory
    """
    
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
        self.polymer_components: Dict[str, SimplePolymerComponent] = {}
        self.box_builder_type = BoxBuilderType.PACKMOL
        self._polymer_builder = None
        
        # TransportProtocol attributes
        self.components = None

    def run_protocol(self):
        """
        Run the polymer transport protocol using the SAME workflow as liquid electrolytes.
        
        Workflow: Build system → NPT → rescale → NVT → (optional) nonequ
        This mirrors TransportProtocol.run_protocol() exactly.
        """
        logger.info("Running Polymer Transport Protocol")

        # IMPORTANT: Generate representative oligomer SMILES for parameterization FIRST,
        # before build_polymer_chains() which would overwrite with full chain SMILES.
        # _get_all_components() sets polymer_comp.smiles to a small trimer SMILES.
        smiles_dict = {}
        for comp in self._get_all_components():
            smiles_dict[comp.name] = comp.smiles
        
        # Now build full polymer chains for box construction (PDB files)
        # This no longer overwrites polymer.smiles
        if self.polymer_components:
            struct_dir = os.path.join(self.output_dir, "structures")
            self.build_polymer_chains(struct_dir)
        
        # Generate force field parameters first (creates .itp, .atp, .gro files)
        nonbonded_params = self.generate_ff_params_polymer(smiles_dict, force=bool(self.config.get('force_regenerate_params', False)))
        
        # # Save nonbonded params for equilibration to use
        # params_json_path = os.path.join(self.output_dir, 'nonbonded_params.json')
        # with open(params_json_path, 'w') as f:
        #     json.dump(nonbonded_params, f, indent=2)
        
        # Build components ratio dict for build_system
        components_ratio = {}
        for comp in self._get_all_components():
            components_ratio[comp.name] = comp.molar_num

        # Use a separate working directory for build_system to avoid SameFileError
        # (build_system copies files from params_dir -> working_dir; they must differ)
        build_working_dir = self.config.get('working_dir', os.path.join(self.output_dir, 'build'))
        
        # Use base Protocol's build_system which creates system.top and solvent_salt.gro
        natoms = self.config.get("natoms", 5000)
        self.components = self.build_system(
            total_atoms=natoms,
            components_ratio=components_ratio,
            # working_dir=self.params_dir,
            working_dir=build_working_dir,
            reuse_if_exists=bool(self.config.get('resume', False)),
        )
        
        # Load system for OpenMM
        gro_file = os.path.join(self.params_dir, "solvent_salt.gro")
        top_file = os.path.join(self.params_dir, "system.top")
        
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        
        input_top, input_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )
        
        # Read timestep and step configuration (same as TransportProtocol)
        def steps_from_time(cfg, steps_key, default_steps, time_ns_key=None, time_ps_key=None, timestep_fs=2):
            if steps_key in cfg:
                return int(cfg[steps_key])
            if time_ns_key and time_ns_key in cfg:
                return int(cfg[time_ns_key] * 1e6 / timestep_fs)
            if time_ps_key and time_ps_key in cfg:
                return int(cfg[time_ps_key] * 1e3 / timestep_fs)
            return default_steps
        
        npt_timestep_fs = int(self.config.get('npt_timestep_fs', 2))
        nvt_timestep_fs = int(self.config.get('nvt_timestep_fs', 2))
        nonequ_timestep_fs = int(self.config.get('nonequ_timestep_fs', 1))
        
        npt_steps = steps_from_time(self.config, 'npt_steps', 4000000, 
                                    time_ns_key='npt_time_ns', time_ps_key='npt_time_ps', 
                                    timestep_fs=npt_timestep_fs)
        nvt_steps = steps_from_time(self.config, 'nvt_steps', 10000000,
                                    time_ns_key='nvt_time_ns', time_ps_key='nvt_time_ps',
                                    timestep_fs=nvt_timestep_fs)
        nonequ_steps = steps_from_time(self.config, 'nonequ_steps', 1000000,
                                       time_ns_key='nonequ_time_ns', time_ps_key='nonequ_time_ps',
                                       timestep_fs=nonequ_timestep_fs)
        
        resume = bool(self.config.get('resume', False))
        checkpoint_interval = int(self.config.get('checkpoint_interval', 5000))
        compute_viscosity = bool(self.config.get('compute_viscosity', True))
        
        # Determine starting stage
        start_from = self.config.get('start_from', 'npt').lower()
        if start_from not in ('npt', 'nvt', 'nonequ'):
            start_from = 'npt'
        
        # Auto-detect stage if resuming
        if resume and self.config.get('start_from') is None:
            nonequ_csv = os.path.join(self.output_dir, 'nonequ_state.csv')
            nvt_csv = os.path.join(self.output_dir, 'nvt_state.csv')
            npt_csv = os.path.join(self.output_dir, 'npt_state.csv')
            
            if os.path.isfile(nonequ_csv):
                start_from = 'nonequ'
            elif os.path.isfile(nvt_csv):
                start_from = 'nvt'
            elif os.path.isfile(npt_csv):
                start_from = 'npt'
        
        logger.info(f'Starting from stage: {start_from}')
        
        # Run stages (same flow as TransportProtocol)
        if start_from == 'npt':
            logger.info('NPT run')
            npt_positions, npt_box_vec = npt_run(
                input_top,
                input_system,
                input_positions,
                temperature=self.config['temperature'],
                npt_steps=npt_steps,
                work_dir=self.output_dir,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                timestep=npt_timestep_fs,
                state_csv_override=self.config.get('npt_state_csv'),
                dcd_path_override=self.config.get('npt_dcd'),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)),
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)),
            )
            
            # Rescale box
            npt_csv_override = self.config.get('npt_state_csv')
            rescale_positions, rescale_box_vec = rescale_box(
                npt_positions,
                npt_box_vec,
                # input_positions,
                # unit_cell,
                work_dir=self.output_dir,
                csv_override=npt_csv_override,
            )
            
            logger.info('NVT run')
            nvt_positions, nvt_box_vec = nvt_run(
                input_top,
                input_system,
                rescale_positions,
                rescale_box_vec,
                temperature=self.config['temperature'],
                work_dir=self.output_dir,
                nvt_steps=nvt_steps,
                timestep=nvt_timestep_fs,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                state_csv_override=self.config.get('nvt_state_csv'),
                dcd_path_override=self.config.get('nvt_dcd'),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)),
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)),
            )
            
        elif start_from == 'nvt':
            # Load NPT state for box rescaling
            npt_csv_override = self.config.get('npt_state_csv')
            npt_csv_path = npt_csv_override or os.path.join(self.output_dir, 'npt_state.csv')
            # if npt_csv_override and os.path.isfile(npt_csv_override):
            #     nvt_seed_pos, nvt_seed_box = rescale_box(
            #         input_positions, unit_cell, 
            #         work_dir=self.output_dir, csv_override=npt_csv_override
            #     )
            if os.path.isfile(npt_csv_path):
                nvt_seed_pos, nvt_seed_box = rescale_box(
                    input_positions, unit_cell, 
                    work_dir=self.output_dir, csv_override=npt_csv_override
                )
            else:
                nvt_seed_pos, nvt_seed_box = input_positions, unit_cell
                logger.info('NPT state not found; using GRO positions/box for NVT')
            
            nvt_positions, nvt_box_vec = nvt_run(
                input_top,
                input_system,
                nvt_seed_pos,
                nvt_seed_box,
                temperature=self.config['temperature'],
                work_dir=self.output_dir,
                nvt_steps=nvt_steps,
                timestep=nvt_timestep_fs,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                state_csv_override=self.config.get('nvt_state_csv'),
                dcd_path_override=self.config.get('nvt_dcd'),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)),
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)),
            )
            
        else:  # start_from == 'nonequ'
            # Load NVT final state
            nvt_dcd = self.config.get('nvt_dcd') or os.path.join(self.output_dir, 'nvt.dcd')
            nvt_csv = self.config.get('nvt_state_csv') or os.path.join(self.output_dir, 'nvt_state.csv')
            
            last = dcd_read(nvt_dcd)[-1]
            nvt_positions = [Vec3(x, y, z) * ou.nanometers for x, y, z in last]
            
            import pandas as pd
            df = pd.read_csv(nvt_csv)
            L = df['Box Volume (nm^3)'].iloc[-1]**(1 / 3)
            nvt_box_vec = (
                Vec3(L, 0.0, 0.0) * ou.nanometers,
                Vec3(0.0, L, 0.0) * ou.nanometers,
                Vec3(0.0, 0.0, L) * ou.nanometers
            )
        
        # Run non-equilibrium MD for viscosity (if requested)
        if compute_viscosity:
            logger.info('Nonequ run')
            nonequ_run(
                input_top,
                input_system,
                nvt_positions,
                nvt_box_vec,
                temperature=self.config['temperature'],
                work_dir=self.output_dir,
                nonequ_steps=nonequ_steps,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                timestep_fs=nonequ_timestep_fs,
            )
        else:
            logger.info('Skipping nonequ run (compute_viscosity=False)')

    # Keep the old run() method as an alias for backward compatibility
    def run(self):
        """Alias for run_protocol() for backward compatibility."""
        self.run_protocol()

    # def _get_all_components(self) -> List[SimpleComponent]:
    #     """Get all components including polymers."""
    #     components: List[SimpleComponent] = []
        
    #     # First add polymer components
    #     for name, polymer_comp in self.polymer_components.items():
    #         # For polymers, we need to generate the full SMILES from monomer
    #         smiles = polymer_comp.smiles
            
    #         # Check if we need to generate SMILES from monomer
    #         if (not smiles or smiles == "") and polymer_comp.monomer_smiles:
    #             # Build a representative SMILES for the polymer
    #             # For parameterization, we use a trimer (3 repeat units) as representative
    #             try:
    #                 from byteff2.toolkit.polymer_builder import PolymerChainBuilder
                    
    #                 # Determine end groups
    #                 left_cap = polymer_comp.end_groups[0] if polymer_comp.end_groups else None
    #                 right_cap = polymer_comp.end_groups[1] if polymer_comp.end_groups else None
                    
    #                 # Use smaller DP for parameterization (full chain would be too large)
    #                 param_dp = min(3, polymer_comp.degree_of_polymerization)
                    
    #                 builder = PolymerChainBuilder(
    #                     monomer_smiles=polymer_comp.monomer_smiles,
    #                     dp=param_dp,
    #                     end_group_left=left_cap,
    #                     end_group_right=right_cap,
    #                     tacticity=polymer_comp.tacticity,
    #                 )
    #                 smiles = builder.get_polymer_smiles()
    #                 polymer_comp.smiles = smiles
    #                 logger.info(f"Generated polymer SMILES for {name}: {smiles}")
    #             except Exception as e:
    #                 logger.warning(f"Could not generate polymer SMILES for {name}: {e}")
    #                 # Fallback: use monomer SMILES directly (remove connection points)
    #                 smiles = polymer_comp.monomer_smiles.replace('[*]', '')
    #                 # If monomer is simple like "CCO", use it directly
    #                 if not smiles:
    #                     smiles = polymer_comp.monomer_smiles
    #                 polymer_comp.smiles = smiles
    #                 logger.info(f"Using fallback SMILES for {name}: {smiles}")
            
    #         if smiles:  # Only add if we have a valid SMILES
    #             components.append(SimpleComponent(
    #                 name=name,
    #                 smiles=smiles,
    #                 type=polymer_comp.type if hasattr(polymer_comp, 'type') else ComponentType.SOLVENT,
    #                 molar_num=polymer_comp.molar_num if hasattr(polymer_comp, 'molar_num') else 1,
    #             ))
    #         else:
    #             logger.warning(f"Skipping polymer component {name}: no SMILES available")
        
    #     # Add non-polymer components from config
    #     for name, comp in self.config.get("components", {}).items():
    #         if isinstance(comp, dict) and comp.get("type") != "polymer":
    #             smiles = self.config.get("smiles", {}).get(name, "")
    #             if smiles:  # Only add if SMILES is available
    #                 comp_type = comp.get("type", "")
    #                 if "cation" in comp_type:
    #                     component_type = ComponentType.CATION
    #                 elif "anion" in comp_type:
    #                     component_type = ComponentType.ANION
    #                 else:
    #                     component_type = ComponentType.SOLVENT
                    
    #                 components.append(SimpleComponent(
    #                     name=name,
    #                     smiles=smiles,
    #                     type=component_type,
    #                     molar_num=comp.get("count", 1),
    #                 ))
        
    #     logger.info(f"Total components: {[c.name for c in components]}")
    #     return components
    
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
        
    # def _run_polymer_equilibration(self):
    #     """Run multi-stage equilibration for polymer system."""
    #     from byteff2.toolkit.polymer_simulation import PolymerEquilibrationProtocol
        
    #     equil_protocol = PolymerEquilibrationProtocol(
    #         temperature=self.config.get("temperature", 300),
    #         pressure=self.config.get("pressure", 1.0),
    #         target_density=self.config.get("target_density"),
    #         stages=self.config.get("equilibration_stages"),
    #         output_dir=self.output_dir,
    #     )
    #     equil_protocol.run_equilibration(
    #         topology_path=os.path.join(self.output_dir, "topol.top"),
    #         coordinates_path=os.path.join(self.output_dir, "system.gro"),
    #         mdp_template_dir=self.output_dir,
    #     )
        
    def post_process(self):
        """
        Post-process transport results using the same approach as liquid electrolytes,
        with optional polymer-specific analyses.
        """
        logger.info('Post-processing polymer transport protocol')
        
        compute_viscosity = bool(self.config.get('compute_viscosity', True))
        compute_conductivity = bool(self.config.get('compute_conductivity', True))
        
        results = {}
        vis = None
        
        if compute_viscosity:
            try:
                vis = viscosity_calc(self.output_dir)
                results['viscosity'] = vis
            except Exception as e:
                logger.warning(f'Viscosity calculation failed: {e}')
        
        if compute_conductivity:
            # Load NVT trajectory
            dcd_path = self.config.get('nvt_dcd') or os.path.join(self.output_dir, 'nvt.dcd')
            if not os.path.isfile(dcd_path):
                dcd_path = 'nvt.dcd'
            
            nvt_positions = dcd_read(dcd_path)
            md_volume, md_temperature = volume_calc(
                self.output_dir, 
                csv_override=self.config.get('nvt_state_csv')
            )
            
            # Build species dictionaries from components
            species_mass_dict, species_number_dict, species_charges_dict = {}, {}, {}
            solvent, cation, anion = [], [], []
            
            for mol_name, topo_mol in self.components.items():
                mass = sum([atom.mass for atom in topo_mol.atoms])
                species_mass_dict[mol_name] = mass
                species_number_dict[mol_name] = topo_mol.molar_num
                species_charges_dict[mol_name] = sum([atom.charge for atom in topo_mol.atoms])
                
                if topo_mol.type == ComponentType.CATION:
                    cation.append(mol_name)
                elif topo_mol.type == ComponentType.ANION:
                    anion.append(mol_name)
                else:
                    solvent.append(mol_name)
            
            # MSD analysis parameters
            skip_frames = int(self.config.get('msd_skip_frames', 100))
            fw_frames = self.config.get('fit_window_frames')
            fw_frac = self.config.get('fit_window_frac', (0.5, 0.9))
            if isinstance(fw_frac, (list, tuple)) and len(fw_frac) == 2:
                fw_frac = tuple(fw_frac)
            else:
                fw_frac = (0.5, 0.9)
            
            output_transference = bool(self.config.get('output_transference', False))
            
            cond = onsager_calc(
                species_mass_dict,
                species_number_dict,
                species_charges_dict,
                md_volume,
                vis,
                md_temperature,
                nvt_positions,
                msd_skip_frames=skip_frames,
                fit_window_frames=fw_frames,
                fit_window_frac=fw_frac,
                compute_transference=output_transference,
            )
            results.update(cond)
        
        # Run polymer-specific analyses if requested
        if self.config.get('polymer_analyses', True):
            polymer_results = self._run_polymer_specific_analyses()
            results['polymer_analysis'] = polymer_results
        
        if results:
            with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        logger.info('Post-processing complete')
        return results
    
    # def _run_transport_production(self):
    #     """Run NVT production for transport property calculation."""
    #     pass
        
    # def post_process_polymer(self):
    #     """Extended post-processing for polymer systems."""
    #     results = {}
        
    #     # Standard transport analysis
    #     if self.config.get("compute_conductivity", True):
    #         results["conductivity"] = self._compute_ionic_conductivity()
            
    #     if self.config.get("compute_viscosity", True):
    #         results["viscosity"] = self._compute_viscosity()
            
    #     # Polymer-specific analyses
    #     if self.config.get("compute_chain_diffusion", True):
    #         results["chain_diffusion"] = self._compute_chain_diffusion()
            
    #     if self.config.get("compute_ion_coordination", True):
    #         results["ion_coordination"] = self._compute_ion_coordination()
            
    #     if self.config.get("compute_ion_hopping", False):
    #         results["ion_hopping"] = self._compute_ion_hopping_statistics()
            
    #     # Save results
    #     results_file = os.path.join(self.output_dir, "polymer_transport_results.json")
    #     with open(results_file, 'w') as f:
    #         json.dump(results, f, indent=2)
            
    #     logger.info(f"Polymer transport results saved to {results_file}")
    #     return results
    
    # def _compute_ionic_conductivity(self):
    #     """Compute ionic conductivity from trajectory."""
    #     # Placeholder - would use existing transport analysis
    #     return None
    
    # def _compute_viscosity(self):
    #     """Compute viscosity from non-equilibrium MD."""
    #     # Placeholder - would use existing viscosity analysis
    #     return None
    
    # def _compute_chain_diffusion(self):
    #     """
    #     Compute polymer chain center-of-mass diffusion.
        
    #     Returns:
    #         Dictionary with chain diffusion coefficients and MSD data
    #     """
    #     logger.info("Computing polymer chain diffusion")
    #     # Placeholder for chain COM MSD analysis
    #     return {"D_chain": None, "msd_data": None}
    
    # def _compute_ion_coordination(self):
    #     """
    #     Analyze ion-polymer coordination over trajectory.
        
    #     Returns:
    #         Dictionary with coordination number statistics
    #     """
    #     logger.info("Computing ion-polymer coordination")
    #     # Placeholder for coordination analysis
    #     return {"avg_coordination": None, "coordination_histogram": None}
    
    # def _compute_ion_hopping_statistics(self):
    #     """
    #     Compute ion hopping statistics between coordination sites.
        
    #     Returns:
    #         Dictionary with hopping rates and residence times
    #     """
    #     logger.info("Computing ion hopping statistics")
    #     # Placeholder for hopping analysis
    #     return {"hopping_rate": None, "residence_time": None}

    # def run_full_workflow(self):
    #     """Run the complete polymer transport workflow."""
    #     logger.info("Running Polymer Transport Protocol - Full Workflow")
    #     self.run()

    def _run_polymer_specific_analyses(self) -> dict:
        """
        Run polymer-specific analyses (optional, does not affect main workflow).
        """
        results = {}
        
        try:
            import MDAnalysis as mda
        except ImportError:
            logger.info("MDAnalysis not available, skipping polymer-specific analyses")
            return results
        
        traj_path = os.path.join(self.output_dir, 'nvt.dcd')
        top_path = os.path.join(self.params_dir, 'solvent_salt.gro')
        
        if not os.path.exists(traj_path) or not os.path.exists(top_path):
            logger.info("Trajectory or topology not found for polymer analysis")
            return results
        
        try:
            u = mda.Universe(top_path, traj_path)
            
            # Chain diffusion analysis
            polymer_selection = u.select_atoms("resname PEO or resname PPO or resname POLY*")
            if len(polymer_selection) > 0:
                from MDAnalysis.analysis.msd import EinsteinMSD
                msd = EinsteinMSD(u, select=polymer_selection, msd_type='xyz', fft=True)
                msd.run()
                
                time = msd.times
                msd_values = msd.results.timeseries
                
                if len(time) > 10:
                    start_idx = len(time) // 2
                    import numpy as np
                    slope, _ = np.polyfit(time[start_idx:], msd_values[start_idx:], 1)
                    D_chain = slope / 6.0 * 1e-5  # Convert to cm²/s
                    results['chain_diffusion_cm2_s'] = float(D_chain)
                    logger.info(f"Polymer chain diffusion: {D_chain:.2e} cm²/s")
            
        except Exception as e:
            logger.warning(f"Polymer analysis failed: {e}")
        
        return results

    def run_full_workflow(self):
        """
        Run the complete polymer transport workflow.
        
        This is the main entry point that follows the liquid electrolyte pattern.
        """
        logger.info("Starting full polymer transport workflow")
        self.run_protocol()
        self.post_process()
        logger.info("Polymer transport workflow complete")

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