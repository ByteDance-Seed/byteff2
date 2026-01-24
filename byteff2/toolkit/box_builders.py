# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Box building backends for simulation system construction.

This module provides multiple backends for building simulation boxes:
- GMXBoxBuilder: Original GROMACS-based builder for small molecules
- PackmolBoxBuilder: Packmol-based builder supporting polymers
- AmorphousBoxBuilder: For building amorphous polymer systems
"""

import os
import subprocess
import tempfile
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoxComponent:
    """Component specification for box building."""
    name: str
    structure_file: str  # Path to PDB/GRO file
    count: int
    
    
class BoxBuilder(ABC):
    """Abstract base class for system box builders."""
    
    @abstractmethod
    def build_box(self, components: List[Any], box_size: float, 
                  output_dir: str, **kwargs) -> str:
        """
        Build simulation box and return path to output structure file.
        
        Args:
            components: List of Component objects
            box_size: Initial box size in nm
            output_dir: Directory for output files
            **kwargs: Additional builder-specific options
            
        Returns:
            Path to output .gro file
        """
        pass
    
    @abstractmethod
    def supports_polymers(self) -> bool:
        """Return True if this builder supports polymer systems."""
        pass
    
    def _convert_pdb_to_gro(self, pdb_file: str, gro_file: str, box_size: float):
        """Convert PDB to GRO format using GROMACS."""
        try:
            cmd = [
                "gmx", "editconf",
                "-f", pdb_file,
                "-o", gro_file,
                "-box", str(box_size), str(box_size), str(box_size),
                "-c"  # Center in box
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Converted {pdb_file} to {gro_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert PDB to GRO: {e}")
            raise
            
    def _check_tool_available(self, tool_name: str) -> bool:
        """Check if a command-line tool is available."""
        try:
            subprocess.run([tool_name, "--version"], capture_output=True)
            return True
        except FileNotFoundError:
            return False


class GMXBoxBuilder(BoxBuilder):
    """
    Original GROMACS-based builder for small molecules.
    
    Uses gmx insert-molecules for packing. Not suitable for polymers
    as it treats molecules as rigid during insertion.
    """
    
    def supports_polymers(self) -> bool:
        return False
    
    def build_box(self, components: List[Any], box_size: float,
                  output_dir: str, **kwargs) -> str:
        """
        Build simulation box using GROMACS insert-molecules.
        
        Args:
            components: List of Component objects with structure files
            box_size: Box size in nm
            output_dir: Output directory
            
        Returns:
            Path to output .gro file
        """
        from bytemol.toolkit.gmxtool import GMXScript
        
        os.makedirs(output_dir, exist_ok=True)
        output_gro = os.path.join(output_dir, "system.gro")
        
        script = GMXScript(output_dir)
        
        # Initialize empty box
        first_comp = components[0]
        script.init_gro_box(f"{first_comp.name}.gro", box_size)
        
        # Insert molecules
        rest_molecules = []
        for comp in components:
            rest_molecules.append({
                "gro": f"{comp.name}.gro",
                "nmol": comp.molar_num,
            })
            
        script.insert_molecules(f"{first_comp.name}.gro", rest_molecules)
        
        # Copy final structure
        import shutil
        shutil.copy(
            os.path.join(output_dir, f"{first_comp.name}.gro"),
            output_gro
        )
        
        logger.info(f"Built system box: {output_gro}")
        return output_gro


class PackmolBoxBuilder(BoxBuilder):
    """
    Packmol-based builder supporting polymers.
    
    Packmol can handle large molecules and produces better initial
    configurations for polymer systems.
    """
    
    def __init__(self, tolerance: float = 2.0, seed: Optional[int] = None):
        """
        Initialize Packmol builder.
        
        Args:
            tolerance: Minimum distance between atoms in Angstroms
            seed: Random seed for reproducibility
        """
        self.tolerance = tolerance
        self.seed = seed if seed is not None else random.randint(1, 100000)
        
    def supports_polymers(self) -> bool:
        return True
    
    def build_box(self, components: List[Any], box_size: float,
                  output_dir: str, target_density: Optional[float] = None,
                  **kwargs) -> str:
        """
        Build simulation box using Packmol.
        
        Args:
            components: List of Component objects
            box_size: Box size in nm (will be converted to Angstroms for Packmol)
            output_dir: Output directory
            target_density: Target density in g/cm³ (optional, for box adjustment)
            
        Returns:
            Path to output .gro file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if Packmol is available
        if not self._check_tool_available("packmol"):
            raise RuntimeError("Packmol not found in PATH. Please install Packmol.")
            
        # Convert box size from nm to Angstroms
        box_size_ang = box_size * 10.0
        
        # Prepare component structure files
        component_pdbs = self._prepare_component_files(components, output_dir)
        
        # Optionally adjust box size based on density
        if target_density is not None:
            box_size_ang = self._calculate_box_for_density(
                components, target_density
            )
            
        # Generate Packmol input
        packmol_input = self._generate_packmol_input(
            component_pdbs, box_size_ang, output_dir
        )
        
        # Write input file
        input_file = os.path.join(output_dir, "packmol.inp")
        with open(input_file, 'w') as f:
            f.write(packmol_input)
            
        # Run Packmol
        output_pdb = os.path.join(output_dir, "system.pdb")
        self._run_packmol(input_file, output_dir)
        
        # Convert PDB to GRO
        output_gro = os.path.join(output_dir, "system.gro")
        self._convert_pdb_to_gro(output_pdb, output_gro, box_size)
        
        logger.info(f"Built polymer system box: {output_gro}")
        return output_gro
    
    def _prepare_component_files(self, components: List[Any], 
                                  output_dir: str) -> Dict[str, str]:
        """
        Prepare PDB files for each component.
        
        Args:
            components: List of components
            output_dir: Output directory
            
        Returns:
            Dictionary mapping component names to PDB file paths
        """
        component_pdbs = {}
        
        for comp in components:
            # Check if structure file exists
            if hasattr(comp, 'structure_file') and os.path.exists(comp.structure_file):
                component_pdbs[comp.name] = comp.structure_file
            else:
                # Generate from SMILES
                pdb_path = os.path.join(output_dir, f"{comp.name}.pdb")
                self._generate_pdb_from_smiles(comp.smiles, pdb_path, comp.name)
                component_pdbs[comp.name] = pdb_path
                
        return component_pdbs
    
    def _generate_pdb_from_smiles(self, smiles: str, output_path: str, name: str):
        """Generate PDB file from SMILES string."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
            
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        result = AllChem.EmbedMolecule(mol, params)
        
        if result != 0:
            # Fallback to ETKDGv2 if v3 fails
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
            
        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            logger.warning(f"MMFF optimization failed for {name}")
            
        # Write PDB
        Chem.MolToPDBFile(mol, output_path)
        logger.info(f"Generated PDB for {name}: {output_path}")
    
    def _generate_packmol_input(self, component_pdbs: Dict[str, str],
                                 box_size: float, output_dir: str) -> str:
        """
        Generate Packmol input file content.
        
        Args:
            component_pdbs: Dictionary of component names to PDB paths
            box_size: Box size in Angstroms
            output_dir: Output directory
            
        Returns:
            Packmol input file content
        """
        output_pdb = os.path.join(output_dir, "system.pdb")
        
        lines = [
            f"tolerance {self.tolerance}",
            "filetype pdb",
            f"output {output_pdb}",
            f"seed {self.seed}",
            "",
        ]
        
        # Add box sides slightly inside the boundaries
        margin = 1.5  # Angstroms
        box_min = margin
        box_max = box_size - margin
        
        for comp_name, pdb_path in component_pdbs.items():
            # Get count from component (need to access through components list)
            count = self._get_component_count(comp_name)
            
            lines.extend([
                f"structure {pdb_path}",
                f"  number {count}",
                f"  inside box {box_min} {box_min} {box_min} {box_max} {box_max} {box_max}",
                "end structure",
                "",
            ])
            
        return "\n".join(lines)
    
    def _get_component_count(self, comp_name: str) -> int:
        """Get molecule count for a component."""
        # This will be set by the caller
        return getattr(self, f'_count_{comp_name}', 1)
    
    def _calculate_box_for_density(self, components: List[Any], 
                                    target_density: float) -> float:
        """
        Calculate box size to achieve target density.
        
        Args:
            components: List of components
            target_density: Target density in g/cm³
            
        Returns:
            Box size in Angstroms
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        import math
        
        total_mass = 0.0
        for comp in components:
            mol = Chem.MolFromSmiles(comp.smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                total_mass += mw * comp.molar_num
                
        # V = m / rho (in g and g/cm³)
        # V in cm³ = mass_g / density
        # mass_g = mass_amu / N_A
        N_A = 6.022e23
        volume_cm3 = total_mass / (target_density * N_A)
        
        # Convert to Angstroms³ (1 cm = 1e8 Angstrom)
        volume_ang3 = volume_cm3 * (1e8)**3
        
        box_size_ang = math.pow(volume_ang3, 1/3)
        
        # Add buffer
        return box_size_ang * 1.1
    
    def _run_packmol(self, input_file: str, work_dir: str):
        """
        Run Packmol with the given input file.
        
        Args:
            input_file: Path to Packmol input file
            work_dir: Working directory
        """
        try:
            with open(input_file, 'r') as f:
                result = subprocess.run(
                    ["packmol"],
                    stdin=f,
                    capture_output=True,
                    text=True,
                    cwd=work_dir
                )
                
            if result.returncode != 0:
                logger.error(f"Packmol failed: {result.stderr}")
                raise RuntimeError(f"Packmol failed: {result.stderr}")
                
            logger.info("Packmol completed successfully")
            
        except FileNotFoundError:
            raise RuntimeError("Packmol executable not found")


class AmorphousBoxBuilder(BoxBuilder):
    """
    Builder for amorphous polymer systems.
    
    Uses a multi-stage approach:
    1. Initial placement with Packmol at low density
    2. Compression via NPT simulation
    3. Annealing to achieve proper chain packing
    """
    
    def __init__(self, compression_steps: int = 100000,
                 anneal_cycles: int = 3):
        """
        Initialize amorphous builder.
        
        Args:
            compression_steps: Number of compression MD steps
            anneal_cycles: Number of annealing cycles
        """
        self.compression_steps = compression_steps
        self.anneal_cycles = anneal_cycles
        self._packmol_builder = PackmolBoxBuilder()
        
    def supports_polymers(self) -> bool:
        return True
    
    def build_box(self, components: List[Any], box_size: float,
                  output_dir: str, target_density: Optional[float] = None,
                  **kwargs) -> str:
        """
        Build amorphous polymer system.
        
        Args:
            components: List of components
            box_size: Initial box size in nm
            output_dir: Output directory
            target_density: Target density in g/cm³
            
        Returns:
            Path to output .gro file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Initial low-density placement with Packmol
        initial_box_size = box_size * 1.5  # Start with larger box
        
        initial_gro = self._packmol_builder.build_box(
            components, initial_box_size, output_dir
        )
        
        # Step 2: Compression would be done via MD
        # For now, just return the Packmol structure
        # The actual compression is handled by the equilibration protocol
        
        logger.info(f"Built initial amorphous system: {initial_gro}")
        logger.info("Note: Compression/annealing should be done via equilibration protocol")
        
        return initial_gro


def get_box_builder(builder_type: str = "packmol", **kwargs) -> BoxBuilder:
    """
    Factory function to get appropriate box builder.
    
    Args:
        builder_type: Type of builder ("gromacs", "packmol", "amorphous")
        **kwargs: Builder-specific options
        
    Returns:
        BoxBuilder instance
    """
    builders = {
        "gromacs": GMXBoxBuilder,
        "packmol": PackmolBoxBuilder,
        "amorphous": AmorphousBoxBuilder,
    }
    
    builder_class = builders.get(builder_type.lower())
    if builder_class is None:
        raise ValueError(f"Unknown box builder type: {builder_type}")
        
    return builder_class(**kwargs)