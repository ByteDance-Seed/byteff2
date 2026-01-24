# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Polymer chain building utilities for ByteFF2.

This module provides tools for:
- Building polymer chains from monomer SMILES
- Handling different polymer architectures (linear, branched, etc.)
- Fragmenting large polymers for parameterization
"""

import os
import logging
import random
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

logger = logging.getLogger(__name__)


class PolymerChainBuilder:
    """
    Build polymer chains from monomer SMILES.
    
    The monomer SMILES should contain connection points marked with [*] atoms.
    For example:
    - PEO: "[*]CCO[*]" or "[*]OCC[*]"
    - PPO: "[*]OC(C)C[*]"
    
    Attributes:
        monomer_smiles: SMILES string of monomer with [*] connection points
        dp: Degree of polymerization (number of repeat units)
        end_groups: Tuple of (left_cap, right_cap) SMILES
        tacticity: Stereochemistry preference
    """
    
    def __init__(self, monomer_smiles: str, dp: int,
                 end_group_left: Optional[str] = None,
                 end_group_right: Optional[str] = None,
                 tacticity: str = "atactic",
                 random_seed: Optional[int] = None):
        """
        Initialize polymer chain builder.
        
        Args:
            monomer_smiles: Monomer SMILES with [*] connection points
            dp: Degree of polymerization
            end_group_left: SMILES for left chain end (default: H)
            end_group_right: SMILES for right chain end (default: H)
            tacticity: "isotactic", "syndiotactic", or "atactic"
            random_seed: Random seed for reproducible tacticity
        """
        self.monomer_smiles = monomer_smiles
        self.dp = dp
        self.end_group_left = end_group_left
        self.end_group_right = end_group_right
        self.tacticity = tacticity
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            
        # Validate monomer SMILES
        self._validate_monomer()
        
    def _validate_monomer(self):
        """Validate that monomer SMILES has exactly 2 connection points."""
        # Count connection points
        n_conn = self.monomer_smiles.count('[*]')
        if n_conn != 2:
            raise ValueError(
                f"Monomer SMILES must have exactly 2 connection points [*], "
                f"found {n_conn} in {self.monomer_smiles}"
            )
            
    def build_chain(self) -> Chem.Mol:
        """
        Build a polymer chain by connecting monomers.
        
        Returns:
            RDKit Mol object representing the polymer chain
        """
        logger.info(f"Building polymer chain: DP={self.dp}, monomer={self.monomer_smiles}")
        
        # Generate polymer SMILES by repeating monomer
        polymer_smiles = self._generate_polymer_smiles()
        
        # Create RDKit molecule
        mol = Chem.MolFromSmiles(polymer_smiles)
        if mol is None:
            raise ValueError(f"Could not parse polymer SMILES: {polymer_smiles}")
            
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer
        mol = self._generate_3d_conformer(mol)
        
        # Apply tacticity if specified
        if self.tacticity != "atactic":
            mol = self._apply_tacticity(mol)
            
        logger.info(f"Built polymer with {mol.GetNumAtoms()} atoms")
        return mol
    
    def _generate_polymer_smiles(self) -> str:
        """
        Generate full polymer SMILES from monomer.
        
        Returns:
            Full polymer SMILES string
        """
        # Replace [*] with numbered connection points for joining
        monomer = self.monomer_smiles.replace('[*]', '[{}H]', 1)
        monomer = monomer.replace('[*]', '[{}H]', 1)
        
        # For simple polymers, we can use a simpler approach:
        # Remove the [*] markers and join monomers directly
        
        # Get the core monomer by removing connection points
        core = self.monomer_smiles.replace('[*]', '')
        
        # Build polymer by repeating core
        if self.dp == 1:
            polymer_core = core
        else:
            # For longer chains, need to handle connectivity properly
            polymer_core = self._build_connected_chain()
            
        # Add end groups
        if self.end_group_left:
            polymer_core = self.end_group_left + polymer_core
        if self.end_group_right:
            polymer_core = polymer_core + self.end_group_right
            
        return polymer_core
    
    def _build_connected_chain(self) -> str:
        """
        Build connected polymer chain using RDKit reactions.
        
        Returns:
            SMILES of connected polymer
        """
        # Parse monomer - replace [*] with labeled atoms for connection
        # Use isotope labels to mark connection points
        labeled_smiles = self.monomer_smiles.replace('[*]', '[3H]', 1)
        labeled_smiles = labeled_smiles.replace('[*]', '[3H]', 1)
        
        monomer_mol = Chem.MolFromSmiles(labeled_smiles)
        if monomer_mol is None:
            raise ValueError(f"Could not parse labeled monomer: {labeled_smiles}")
            
        # Find the tritium atoms (our connection points)
        tritium_indices = []
        for atom in monomer_mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
                tritium_indices.append(atom.GetIdx())
                
        if len(tritium_indices) != 2:
            raise ValueError(f"Expected 2 connection points, found {len(tritium_indices)}")
            
        # Build chain iteratively
        growing_chain = Chem.RWMol(monomer_mol)
        
        for i in range(self.dp - 1):
            # Add another monomer unit
            new_monomer = Chem.MolFromSmiles(labeled_smiles)
            growing_chain = self._connect_monomers(growing_chain, new_monomer)
            
        # Convert tritium back to hydrogen for end groups
        final_mol = growing_chain.GetMol()
        final_smiles = Chem.MolToSmiles(final_mol)
        
        # Remove isotope labels
        final_smiles = final_smiles.replace('[3H]', '')
        
        return final_smiles
    
    def _connect_monomers(self, mol1: Chem.RWMol, mol2: Chem.Mol) -> Chem.RWMol:
        """
        Connect two monomer units at their connection points.
        
        Args:
            mol1: First molecule (growing chain)
            mol2: Second molecule (new monomer)
            
        Returns:
            Combined molecule
        """
        # Find tritium atoms in mol1 (right end) and mol2 (left end)
        t1_idx = None
        t2_idx = None
        
        for atom in mol1.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
                # Get the rightmost tritium (highest index)
                if t1_idx is None or atom.GetIdx() > t1_idx:
                    t1_idx = atom.GetIdx()
                    
        for atom in mol2.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 3:
                # Get the leftmost tritium (lowest index)
                if t2_idx is None or atom.GetIdx() < t2_idx:
                    t2_idx = atom.GetIdx()
                    
        if t1_idx is None or t2_idx is None:
            raise ValueError("Could not find connection points in monomers")
            
        # Get the atoms connected to the tritiums
        t1_neighbor = mol1.GetAtomWithIdx(t1_idx).GetNeighbors()[0].GetIdx()
        
        # Combine molecules
        combo = Chem.CombineMols(mol1.GetMol(), mol2)
        combo = Chem.RWMol(combo)
        
        # Adjust t2_idx for combined molecule
        offset = mol1.GetNumAtoms()
        t2_idx_combo = t2_idx + offset
        t2_neighbor = combo.GetAtomWithIdx(t2_idx_combo).GetNeighbors()[0].GetIdx()
        
        # Add bond between the neighbors
        combo.AddBond(t1_neighbor, t2_neighbor, Chem.BondType.SINGLE)
        
        # Remove the tritium atoms (in reverse order to maintain indices)
        to_remove = sorted([t1_idx, t2_idx_combo], reverse=True)
        for idx in to_remove:
            combo.RemoveAtom(idx)
            
        return combo
    
    def _generate_3d_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Generate 3D conformer for the polymer.
        
        For long chains, uses a multi-stage approach to avoid
        bad conformations.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Molecule with 3D coordinates
        """
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed if self.random_seed else -1
        params.maxIterations = 5000
        
        # For very long chains, use more attempts
        if mol.GetNumAtoms() > 200:
            params.numThreads = 0  # Use all available threads
            params.useRandomCoords = True
            
        result = AllChem.EmbedMolecule(mol, params)
        
        if result != 0:
            # Fallback to simpler embedding
            logger.warning("ETKDGv3 failed, trying simpler embedding")
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
            
        # Optimize geometry
        try:
            if mol.GetNumAtoms() < 500:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            else:
                # For large molecules, use UFF which is faster
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception as e:
            logger.warning(f"Force field optimization failed: {e}")
            
        return mol
    
    def _apply_tacticity(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Apply tacticity to the polymer chain.
        
        Args:
            mol: Polymer molecule
            
        Returns:
            Molecule with applied tacticity
        """
        # Find stereocenters
        stereo_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        
        if not stereo_info:
            return mol
            
        rw_mol = Chem.RWMol(mol)
        
        for idx, chirality in stereo_info:
            if chirality == '?':  # Unassigned
                if self.tacticity == "isotactic":
                    # All same configuration
                    rw_mol.GetAtomWithIdx(idx).SetChiralTag(
                        Chem.ChiralType.CHI_TETRAHEDRAL_CW
                    )
                elif self.tacticity == "syndiotactic":
                    # Alternating configuration
                    if idx % 2 == 0:
                        rw_mol.GetAtomWithIdx(idx).SetChiralTag(
                            Chem.ChiralType.CHI_TETRAHEDRAL_CW
                        )
                    else:
                        rw_mol.GetAtomWithIdx(idx).SetChiralTag(
                            Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                        )
                        
        return rw_mol.GetMol()
    
    def get_polymer_smiles(self) -> str:
        """
        Get the SMILES string of the built polymer.
        
        Returns:
            Polymer SMILES string
        """
        return self._generate_polymer_smiles()
    
    def save_pdb(self, mol: Chem.Mol, output_path: str, 
                 resname: str = "MOL"):
        """
        Save polymer to PDB file.
        
        Args:
            mol: RDKit molecule with 3D coordinates
            output_path: Output file path
            resname: Residue name (max 3 characters)
        """
        # Set residue info
        for atom in mol.GetAtoms():
            mi = Chem.AtomPDBResidueInfo()
            mi.SetResidueName(resname[:3])
            mi.SetResidueNumber(1)
            mi.SetIsHeteroAtom(True)
            atom.SetPDBResidueInfo(mi)
            
        Chem.MolToPDBFile(mol, output_path)
        logger.info(f"Saved polymer to {output_path}")
        
    def generate_mapped_smiles(self) -> str:
        """
        Generate atom-mapped SMILES for the full polymer chain.
        
        This is useful for force field parameterization.
        
        Returns:
            Atom-mapped SMILES string
        """
        mol = Chem.MolFromSmiles(self._generate_polymer_smiles())
        if mol is None:
            return ""
            
        # Add atom mapping
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
            
        return Chem.MolToSmiles(mol)


class PEOBuilder(PolymerChainBuilder):
    """Specialized builder for Poly(ethylene oxide)."""
    
    def __init__(self, dp: int, **kwargs):
        super().__init__(
            monomer_smiles="[*]OCC[*]",
            dp=dp,
            end_group_left=kwargs.get("end_group_left", "C"),
            end_group_right=kwargs.get("end_group_right", "O"),
            **{k: v for k, v in kwargs.items() 
               if k not in ["end_group_left", "end_group_right"]}
        )


class PPOBuilder(PolymerChainBuilder):
    """Specialized builder for Poly(propylene oxide)."""
    
    def __init__(self, dp: int, **kwargs):
        super().__init__(
            monomer_smiles="[*]OC(C)C[*]",
            dp=dp,
            **kwargs
        )


class PVDFBuilder(PolymerChainBuilder):
    """Specialized builder for Poly(vinylidene fluoride)."""
    
    def __init__(self, dp: int, **kwargs):
        super().__init__(
            monomer_smiles="[*]C(F)(F)C[*]",
            dp=dp,
            **kwargs
        )


class PANBuilder(PolymerChainBuilder):
    """Specialized builder for Poly(acrylonitrile)."""
    
    def __init__(self, dp: int, **kwargs):
        super().__init__(
            monomer_smiles="[*]CC(C#N)[*]",
            dp=dp,
            **kwargs
        )


class PolymerLibrary:
    """Pre-defined polymer builders for common systems."""
    
    POLYMERS = {
        'PEO': PEOBuilder,
        'PPO': PPOBuilder,
        'PVDF': PVDFBuilder,
        'PAN': PANBuilder,
    }
    
    @classmethod
    def get_builder(cls, polymer_name: str, dp: int, **kwargs) -> PolymerChainBuilder:
        """
        Get a polymer builder by name.
        
        Args:
            polymer_name: Name of the polymer (e.g., 'PEO', 'PPO')
            dp: Degree of polymerization
            **kwargs: Additional builder options
            
        Returns:
            PolymerChainBuilder instance
        """
        builder_class = cls.POLYMERS.get(polymer_name.upper())
        if builder_class is None:
            raise ValueError(
                f"Unknown polymer: {polymer_name}. "
                f"Available: {list(cls.POLYMERS.keys())}"
            )
        return builder_class(dp, **kwargs)
    
    @classmethod
    def register_polymer(cls, name: str, monomer_smiles: str,
                        default_left_cap: str = None,
                        default_right_cap: str = None):
        """
        Register a new polymer type.
        
        Args:
            name: Polymer name
            monomer_smiles: Monomer SMILES with [*] connection points
            default_left_cap: Default left end group
            default_right_cap: Default right end group
        """
        def builder_factory(dp: int, **kwargs):
            return PolymerChainBuilder(
                monomer_smiles=monomer_smiles,
                dp=dp,
                end_group_left=kwargs.get("end_group_left", default_left_cap),
                end_group_right=kwargs.get("end_group_right", default_right_cap),
                **{k: v for k, v in kwargs.items() 
                   if k not in ["end_group_left", "end_group_right"]}
            )
        cls.POLYMERS[name.upper()] = builder_factory


def fragment_polymer_for_params(polymer_smiles: str, 
                                 fragment_size: int = 3) -> Tuple[Dict[str, str], Dict[int, Tuple[str, int]]]:
    """
    Fragment a polymer into representative units for parameterization.
    
    This function identifies unique chemical environments in the polymer
    and creates fragments that can be parameterized individually.
    
    Args:
        polymer_smiles: Full polymer SMILES
        fragment_size: Number of repeat units per fragment
        
    Returns:
        Tuple of (fragments dict, atom_mapping dict)
        - fragments: {fragment_name: fragment_smiles}
        - atom_mapping: {polymer_atom_idx: (fragment_name, fragment_atom_idx)}
    """
    mol = Chem.MolFromSmiles(polymer_smiles)
    if mol is None:
        raise ValueError(f"Could not parse polymer SMILES: {polymer_smiles}")
        
    fragments = {}
    atom_mapping = {}
    
    # For simple linear polymers, identify:
    # 1. Left end group fragment
    # 2. Middle repeat unit fragments
    # 3. Right end group fragment
    
    # This is a simplified implementation - a full version would
    # use more sophisticated fragmentation based on chemical environment
    
    num_atoms = mol.GetNumAtoms()
    atoms_per_fragment = min(fragment_size * 10, num_atoms)  # Approximate
    
    # Create overlapping fragments
    fragment_idx = 0
    processed = set()
    
    for start_idx in range(0, num_atoms, atoms_per_fragment // 2):
        end_idx = min(start_idx + atoms_per_fragment, num_atoms)
        
        # Get atom indices for this fragment
        frag_atom_indices = list(range(start_idx, end_idx))
        
        # Create fragment SMILES
        frag_name = f"fragment_{fragment_idx}"
        
        # Get the substructure
        atom_map = {}
        for i, idx in enumerate(frag_atom_indices):
            atom_map[idx] = i
            if idx not in processed:
                atom_mapping[idx] = (frag_name, i)
                processed.add(idx)
                
        # Create fragment molecule
        frag_mol = Chem.RWMol()
        old_to_new = {}
        
        for i, idx in enumerate(frag_atom_indices):
            atom = mol.GetAtomWithIdx(idx)
            new_idx = frag_mol.AddAtom(atom)
            old_to_new[idx] = new_idx
            
        # Add bonds within fragment
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in old_to_new and end_idx in old_to_new:
                frag_mol.AddBond(
                    old_to_new[begin_idx],
                    old_to_new[end_idx],
                    bond.GetBondType()
                )
                
        try:
            Chem.SanitizeMol(frag_mol)
            fragments[frag_name] = Chem.MolToSmiles(frag_mol.GetMol())
        except Exception:
            # If sanitization fails, use a simpler approach
            fragments[frag_name] = polymer_smiles
            
        fragment_idx += 1
        
        if end_idx >= num_atoms:
            break
            
    return fragments, atom_mapping


def build_polymer_from_config(config: dict, output_dir: str) -> Tuple[Chem.Mol, str]:
    """
    Build a polymer from a configuration dictionary.
    
    Args:
        config: Configuration with polymer specifications
        output_dir: Directory for output files
        
    Returns:
        Tuple of (RDKit molecule, output PDB path)
    """
    polymer_name = config.get("name", "polymer")
    
    # Check if using a predefined polymer
    if polymer_name.upper() in PolymerLibrary.POLYMERS:
        builder = PolymerLibrary.get_builder(
            polymer_name,
            dp=config.get("dp", config.get("degree_of_polymerization", 10)),
            end_group_left=config.get("left_cap"),
            end_group_right=config.get("right_cap"),
            tacticity=config.get("tacticity", "atactic"),
        )
    else:
        # Use custom monomer SMILES
        builder = PolymerChainBuilder(
            monomer_smiles=config.get("monomer_smiles", "[*]CCO[*]"),
            dp=config.get("dp", config.get("degree_of_polymerization", 10)),
            end_group_left=config.get("left_cap"),
            end_group_right=config.get("right_cap"),
            tacticity=config.get("tacticity", "atactic"),
        )
        
    # Build the chain
    mol = builder.build_chain()
    
    # Save to PDB
    os.makedirs(output_dir, exist_ok=True)
    resname = config.get("resname", polymer_name[:3].upper())
    output_path = os.path.join(output_dir, f"{polymer_name}.pdb")
    builder.save_pdb(mol, output_path, resname=resname)
    
    return mol, output_path