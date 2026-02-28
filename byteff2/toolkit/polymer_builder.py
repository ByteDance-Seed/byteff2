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
    
    IMPORTANT: The order of [*] in SMILES determines connectivity:
    - First [*] = LEFT attachment point (will connect to previous monomer's RIGHT)
    - Second [*] = RIGHT attachment point (will connect to next monomer's LEFT)

    Attributes:
        monomer_smiles: SMILES string of monomer with [*] connection points
        dp: Degree of polymerization (number of repeat units)
        end_groups: Tuple of (left_cap, right_cap) SMILES
        tacticity: Stereochemistry preference
    """
    
    def __init__(self, monomer_smiles: str, dp: int,
                 end_group_left: Optional[str] = None,
                 end_group_right: Optional[str] = None,
                 tacticity: str = "atactic"):
                #  random_seed: Optional[int] = None):
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
        # self.random_seed = random_seed
        self._polymer_mol = None
        self._polymer_smiles = None
        
        # if random_seed is not None:
        #     random.seed(random_seed)
            
        # # Validate monomer SMILES
        # self._validate_monomer()
        
    def _find_attachment_points(self, mol) -> List[int]:
        """Find indices of dummy atoms ([*]) used as attachment points.
        Returns them in the order they appear in the SMILES string,
        which determines left (first) vs right (second) connectivity.
        """
        attachment_points = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom [*]
                attachment_points.append(atom.GetIdx())
        return attachment_points
    
    def _get_neighbor_of_dummy(self, mol, dummy_idx: int) -> int:
        """Get the index of the real atom bonded to a dummy atom."""
        atom = mol.GetAtomWithIdx(dummy_idx)
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError(
                f"Dummy atom at index {dummy_idx} has {len(neighbors)} neighbors; expected 1"
            )
        return neighbors[0].GetIdx()
    
    def _validate_monomer(self):
        """Validate that monomer SMILES has exactly 2 connection points."""
        # Count connection points
        n_conn = self.monomer_smiles.count('[*]')
        if n_conn != 2:
            raise ValueError(
                f"Monomer SMILES must have exactly 2 connection points [*], "
                f"found {n_conn} in {self.monomer_smiles}"
            )
            
    # def build_chain(self) -> Chem.Mol:
    #     """
    #     Build a polymer chain by connecting monomers.
        
    #     Uses CombineMols + RWMol to iteratively join monomer units
    #     instead of SMILES string concatenation.
        
    #     Returns:
    #         RDKit Mol object of the full polymer chain
    #     """
    #     monomer = Chem.MolFromSmiles(self.monomer_smiles)
    #     if monomer is None:
    #         raise ValueError(f"Could not parse monomer SMILES: {self.monomer_smiles}")
        
    #     attachment_points = self._find_attachment_points(monomer)
    #     if len(attachment_points) != 2:
    #         raise ValueError(
    #             f"Monomer must have exactly 2 attachment points [*], "
    #             f"found {len(attachment_points)} in '{self.monomer_smiles}'"
    #         )
        
    #     logger.info(f"Building polymer chain: DP={self.dp}, monomer={self.monomer_smiles}")
        
    #     # Build using RDKit molecular editing (avoids SMILES nesting limit)
    #     polymer = self._build_chain_rwmol(monomer, attachment_points)
        
    #     if polymer is None:
    #         raise ValueError(f"Failed to build polymer chain with DP={self.dp}")
        
    #     # Add explicit Hs for proper 3D embedding
    #     polymer = Chem.AddHs(polymer)

    #     # Generate 3D coordinates
    #     try:
    #         params = AllChem.ETKDGv3()
    #         params.useRandomCoords = True  # Better for large molecules
    #         params.maxIterations = 5000
    #         result = AllChem.EmbedMolecule(polymer, params)
    #         if result == 0:
    #             try:
    #                 AllChem.MMFFOptimizeMolecule(polymer, maxIters=500)
    #             except Exception:
    #                 pass  # Optimization failure is non-fatal
    #         else:
    #             raise RuntimeError(f"EmbedMolecule returned {result}")
    #     except Exception as e:
    #         logger.warning(f"3D coordinate generation failed, trying fallback: {e}")
    #         try:
    #             params2 = AllChem.ETKDGv2()
    #             params2.useRandomCoords = True
    #             params2.maxIterations = 10000
    #             result = AllChem.EmbedMolecule(polymer, params2)
    #             if result != 0:
    #                 logger.warning("ETKDGv2 also failed, using random coordinates")
    #                 AllChem.EmbedMolecule(polymer, useRandomCoords=True)
    #         except Exception as e2:
    #             logger.warning(f"All 3D embedding failed: {e2}")
    #             AllChem.Compute2DCoords(polymer)
        
    #     # Remove explicit Hs to keep atom count consistent with SMILES
    #     polymer = Chem.RemoveHs(polymer)
        
    #     self._polymer_mol = polymer
    #     self._polymer_smiles = Chem.MolToSmiles(polymer)
        
    #     logger.info(f"Built polymer with {polymer.GetNumAtoms()} atoms")
    #     return polymer

    def build_chain(self) -> Chem.Mol:
        """
        Build a polymer chain by connecting monomers.
        
        Uses CombineMols + RWMol to iteratively join monomer units
        instead of SMILES string concatenation.
        
        Returns:
            RDKit Mol object of the full polymer chain
        """
        monomer = Chem.MolFromSmiles(self.monomer_smiles)
        if monomer is None:
            raise ValueError(f"Could not parse monomer SMILES: {self.monomer_smiles}")
        
        attachment_points = self._find_attachment_points(monomer)
        if len(attachment_points) != 2:
            raise ValueError(
                f"Monomer must have exactly 2 attachment points [*], "
                f"found {len(attachment_points)} in '{self.monomer_smiles}'"
            )
        
        logger.info(f"Building polymer chain: DP={self.dp}, monomer={self.monomer_smiles}")
        
        # Build using RDKit molecular editing (avoids SMILES nesting limit)
        polymer = self._build_chain_rwmol(monomer, attachment_points)
        
        if polymer is None:
            raise ValueError(f"Failed to build polymer chain with DP={self.dp}")
        
        # Add explicit Hs for proper 3D embedding
        polymer = Chem.AddHs(polymer)

        # Use staged embedding for reliable 3D coordinates
        polymer = self._staged_embed(polymer)
        
        # Remove explicit Hs to keep atom count consistent with SMILES
        polymer = Chem.RemoveHs(polymer)
        
        self._polymer_mol = polymer
        self._polymer_smiles = Chem.MolToSmiles(polymer)
        
        logger.info(f"Built polymer with {polymer.GetNumAtoms()} atoms")
        return polymer

    def _staged_embed(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Generate 3D coordinates using a staged approach for large molecules.
        
        For small molecules (< 200 atoms), use standard ETKDGv3.
        For larger molecules, use a constrained embedding approach:
        1. Embed the molecule with random coordinates
        2. Run UFF optimization in stages with increasing force field iterations
        3. Verify bond lengths are reasonable
        
        Args:
            mol: RDKit molecule with explicit Hs
            
        Returns:
            Molecule with optimized 3D coordinates
        """
        natoms = mol.GetNumAtoms()
        
        if natoms < 200:
            # Small molecule: standard approach works fine
            try:
                params = AllChem.ETKDGv3()
                params.useRandomCoords = False
                params.maxIterations = 5000
                result = AllChem.EmbedMolecule(mol, params)
                if result == 0:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                    except Exception:
                        pass
                    if self._check_bond_lengths(mol):
                        return mol
            except Exception:
                pass
            # Fallback for small molecules
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
                AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
                return mol
            except Exception:
                AllChem.Compute2DCoords(mol)
                return mol
        
        # Large molecule: staged approach
        logger.info(f"Using staged embedding for large molecule ({natoms} atoms)")
        
        # Stage 1: Embed with random coordinates (always succeeds for large mols)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.maxIterations = 0  # Don't optimize during embedding
        params.randomSeed = 42
        # Use a larger bounding box to avoid initial clashes
        params.boxSizeMult = 3.0
        params.useBasicKnowledge = True
        params.enforceChirality = False  # Relax for initial embed
        
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            # Last resort: pure random coordinates
            logger.warning("ETKDGv3 random failed; using bare random coords")
            AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=50)
        
        if mol.GetNumConformers() == 0:
            logger.error("Could not embed molecule at all; using 2D coords")
            AllChem.Compute2DCoords(mol)
            return mol
        
        # Stage 2: Progressive UFF optimization
        # UFF is much faster and more robust than MMFF for large molecules
        logger.info("Stage 2: Progressive UFF optimization")
        for max_iters in [200, 500, 1000, 2000, 5000]:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol)
                if ff is None:
                    logger.warning("UFF force field creation failed; trying MMFF")
                    break
                ff.Initialize()
                converged = ff.Minimize(maxIts=max_iters, energyTol=1e-4, forceTol=1e-3)
                energy = ff.CalcEnergy()
                logger.info(f"  UFF opt ({max_iters} iters): converged={converged}, energy={energy:.1f}")
                if converged == 0:
                    break
            except Exception as e:
                logger.warning(f"  UFF optimization failed at {max_iters} iters: {e}")
                break
        
        # Stage 3: Verify and fix bond lengths
        if not self._check_bond_lengths(mol, tolerance=0.5):
            logger.warning("Bond lengths still poor after UFF; attempting MMFF refinement")
            try:
                result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
                if result == -1:
                    logger.warning("MMFF setup failed")
                else:
                    logger.info(f"MMFF refinement result: {result}")
            except Exception as e:
                logger.warning(f"MMFF refinement failed: {e}")
        
        # Final check
        if self._check_bond_lengths(mol):
            logger.info("Bond length check passed after optimization")
        else:
            logger.warning("Bond lengths may still be non-ideal; proceeding anyway")
        
        return mol
    
    def _check_bond_lengths(self, mol: Chem.Mol, tolerance: float = 0.3) -> bool:
        """
        Check if bond lengths in the conformer are reasonable.
        
        Expected bond lengths (Angstrom):
        - C-C: ~1.54, C=C: ~1.34
        - C-O: ~1.43, C=O: ~1.23
        - C-H: ~1.09
        - C-F: ~1.35
        - O-H: ~0.97
        - C-N: ~1.47
        
        Args:
            mol: RDKit molecule with at least one conformer
            tolerance: Maximum allowed deviation from ideal (Angstrom)
            
        Returns:
            True if all bond lengths are within tolerance
        """
        if mol.GetNumConformers() == 0:
            return False
        
        # Reference bond lengths (Angstrom)
        TYPICAL_BOND_LENGTHS = {
            (1, 6): 1.09,   # H-C
            (1, 7): 1.01,   # H-N
            (1, 8): 0.97,   # H-O
            (6, 6): 1.54,   # C-C
            (6, 7): 1.47,   # C-N
            (6, 8): 1.43,   # C-O
            (6, 9): 1.35,   # C-F
            (6, 16): 1.82,  # C-S
            (6, 17): 1.77,  # C-Cl
            (7, 8): 1.36,   # N-O (amide-like)
            (7, 16): 1.65,  # N-S
            (8, 16): 1.50,  # O-S (in TFSI)
        }
        
        conf = mol.GetConformer(0)
        n_bad = 0
        n_total = 0
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = pos_i.Distance(pos_j)
            
            n1 = mol.GetAtomWithIdx(i).GetAtomicNum()
            n2 = mol.GetAtomWithIdx(j).GetAtomicNum()
            key = tuple(sorted([n1, n2]))
            
            expected = TYPICAL_BOND_LENGTHS.get(key, 1.5)
            n_total += 1
            
            if abs(dist - expected) > tolerance:
                n_bad += 1
        
        if n_bad > 0:
            bad_frac = n_bad / max(n_total, 1)
            logger.debug(f"Bond length check: {n_bad}/{n_total} bonds ({bad_frac:.1%}) outside tolerance")
            if bad_frac > 0.1:  # More than 10% bad bonds
                return False
        
        return True

    def _build_chain_rwmol(self, monomer: Chem.Mol, attachment_points: List[int]) -> Chem.Mol:
        """
        Build polymer chain by iteratively combining monomer units using RWMol.
        
        The key insight: we connect the NEIGHBOR of the right dummy to the NEIGHBOR
        of the left dummy. For [*]OC(C(C)F)[*]:
        - Left dummy's neighbor is O
        - Right dummy's neighbor is C
        
        So we form: ...-O-[from growing chain] + [C-from new monomer]-...
        Which gives: ...-O-C-... (ether linkage) ✓
        
        NOT: ...-O-[from growing] + [O-from new]-... which would give O-O (peroxy) ✗
        """
    def _build_chain_rwmol(self, monomer: Chem.Mol, attachment_points: List[int]) -> Chem.Mol:
        """
        Build polymer chain by iteratively combining monomer units.
        
        The key insight: we connect the NEIGHBOR of the right dummy to the NEIGHBOR
        of the left dummy. For [*]OC(C(C)F)[*]:
        - Left dummy's neighbor is O
        - Right dummy's neighbor is C
        
        So we form: ...-O-[from growing chain] + [C-from new monomer]-...
        Which gives: ...-O-C-... (ether linkage) ✓
        
        NOT: ...-O-[from growing] + [O-from new]-... which would give O-O (peroxy) ✗
        """
        # Identify left/right dummies and their real neighbors
        left_dummy = attachment_points[0]
        right_dummy = attachment_points[1]
        
        # Get the real atoms bonded to the dummies
        left_neighbor = self._get_neighbor_of_dummy(monomer, left_dummy)
        right_neighbor = self._get_neighbor_of_dummy(monomer, right_dummy)
        
        # Log for debugging
        left_atom = monomer.GetAtomWithIdx(left_neighbor)
        right_atom = monomer.GetAtomWithIdx(right_neighbor)
        logger.debug(f"Monomer: left dummy {left_dummy} -> {left_atom.GetSymbol()}({left_neighbor}), "
                     f"right dummy {right_dummy} -> {right_atom.GetSymbol()}({right_neighbor})")
        
        # Get bond type (usually single)
        bond = monomer.GetBondBetweenAtoms(left_dummy, left_neighbor)
        bond_type = bond.GetBondType() if bond else Chem.BondType.SINGLE
        
        # Start with first monomer
        growing = Chem.RWMol(monomer)
        current_right_dummy = right_dummy
        
        for i in range(1, self.dp):
            # Create fresh monomer
            new_monomer = Chem.MolFromSmiles(self.monomer_smiles)
            new_ap = self._find_attachment_points(new_monomer)
            new_left_dummy = new_ap[0]
            new_right_dummy = new_ap[1]
            
            # Get the real atoms to connect BEFORE combining molecules
            # From growing chain: the atom bonded to the right dummy
            growing_connect_atom = self._get_neighbor_of_dummy(growing, current_right_dummy)
            # From new monomer: the atom bonded to the left dummy
            new_connect_atom = self._get_neighbor_of_dummy(new_monomer, new_left_dummy)
            
            # Combine molecules
            offset = growing.GetNumAtoms()
            combo = Chem.RWMol(Chem.CombineMols(growing, new_monomer))
            
            # Adjust indices for combined molecule
            combo_growing_connect = growing_connect_atom  # unchanged
            combo_new_connect = new_connect_atom + offset
            combo_growing_right_dummy = current_right_dummy
            combo_new_left_dummy = new_left_dummy + offset
            combo_new_right_dummy = new_right_dummy + offset
            
            # Add bond between the real atoms (NOT the dummies)
            # This connects: growing_chain[right_neighbor] - new_monomer[left_neighbor]
            combo.AddBond(combo_growing_connect, combo_new_connect, bond_type)
            
            # Remove the linked dummies (higher index first)
            dummies_to_remove = sorted([combo_growing_right_dummy, combo_new_left_dummy], reverse=True)
            
            # Track index shifts
            new_right_dummy_final = combo_new_right_dummy
            for d_idx in dummies_to_remove:
                combo.RemoveAtom(d_idx)
                if d_idx < new_right_dummy_final:
                    new_right_dummy_final -= 1
            
            current_right_dummy = new_right_dummy_final
            
            try:
                Chem.SanitizeMol(combo)
            except Exception as e:
                logger.warning(f"Sanitization warning at DP={i+1}: {e}")
            
            growing = combo
        
        # Cap the ends
        return self._cap_chain_ends(growing)
    
    def _cap_chain_ends(self, chain: Chem.Mol) -> Chem.Mol:
        """
        Cap the chain ends by replacing remaining dummy atoms with end groups.
        
        If no end groups specified, simply remove dummies and add H.
        """
        rw = Chem.RWMol(chain)
        
        # Find remaining dummy atoms
        dummies = []
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummies.append(atom.GetIdx())
        
        if len(dummies) == 0:
            # No dummies left; chain is already capped
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        
        if self.end_group_left and self.end_group_right and len(dummies) >= 2:
            # Cap with specified end groups using fragment attachment
            mol = self._attach_end_groups(rw, dummies)
        else:
            # Simple capping: remove dummies (hydrogen will be implicit)
            for d_idx in sorted(dummies, reverse=True):
                # Get the neighbor to adjust its implicit H count
                neighbor_idx = self._get_neighbor_of_dummy(rw, d_idx)
                rw.RemoveAtom(d_idx)
                # After removal, neighbor_idx may shift if d_idx < neighbor_idx
                # Recalculate
            
            try:
                mol = rw.GetMol()
                Chem.SanitizeMol(mol)
            except Exception:
                mol = rw.GetMol()
            
        return mol
    
    def _attach_end_groups(self, chain: Chem.RWMol, dummy_indices: List[int]) -> Chem.Mol:
        """Attach end group molecules to chain ends."""
        # Sort dummies: first = left end, last = right end
        dummy_indices = sorted(dummy_indices)
        left_dummy = dummy_indices[0]
        right_dummy = dummy_indices[-1]
        
        left_cap_mol = Chem.MolFromSmiles(self.end_group_left) if self.end_group_left else None
        right_cap_mol = Chem.MolFromSmiles(self.end_group_right) if self.end_group_right else None
        
        # For simple end groups (like "C" for methyl, "O" for hydroxyl),
        # replace dummy with the end group atom
        result = Chem.RWMol(chain)
        
        # Process right end first (higher index, won't affect left index)
        caps = [(right_dummy, right_cap_mol), (left_dummy, left_cap_mol)]
        
        for d_idx, cap_mol in sorted(caps, key=lambda x: x[0], reverse=True):
            if cap_mol is None:
                # Just remove dummy
                result.RemoveAtom(d_idx)
                continue
            
            neighbor_idx = self._get_neighbor_of_dummy(result, d_idx)
            
            if cap_mol.GetNumAtoms() == 1:
                # Simple atom cap: replace dummy with this atom
                cap_atom = cap_mol.GetAtomWithIdx(0)
                result.ReplaceAtom(d_idx, cap_atom)
            else:
                # Multi-atom end group: combine and bond
                # Find attachment point in cap (look for [*] or use first atom)
                cap_attachment = 0
                for atom in cap_mol.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        cap_attachment = atom.GetIdx()
                        break
                
                combo = Chem.RWMol(Chem.CombineMols(result, cap_mol))
                offset = result.GetNumAtoms()
                
                cap_real = offset + cap_attachment
                cap_neighbor = self._get_neighbor_of_dummy(combo, cap_real) if cap_mol.GetAtomWithIdx(cap_attachment).GetAtomicNum() == 0 else cap_real
                
                combo.AddBond(neighbor_idx, cap_neighbor, Chem.BondType.SINGLE)
                
                # Remove dummies
                dummies_to_remove = sorted([d_idx, cap_real] if cap_mol.GetAtomWithIdx(cap_attachment).GetAtomicNum() == 0 else [d_idx], reverse=True)
                for rm_idx in dummies_to_remove:
                    combo.RemoveAtom(rm_idx)
                
                result = combo
        
        try:
            mol = result.GetMol()
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"End group capping sanitization: {e}")
            mol = result.GetMol()
        
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
        """Get SMILES of built polymer. Build if not done."""
        if self._polymer_smiles is None:
            self.build_chain()
        return self._polymer_smiles
    
    def save_pdb(self, mol: Chem.Mol, output_path: str, resname: str = "POL"):
        """Save polymer to PDB file."""
        # Ensure 3D coordinates exist
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        
        Chem.MolToPDBFile(mol, output_path)
        
        # Fix residue name in PDB
        with open(output_path, 'r') as f:
            content = f.read()
        content = content.replace('UNL', resname[:3].ljust(3))
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved polymer to {output_path}")
        
    # def generate_mapped_smiles(self) -> str:
    #     """
    #     Generate atom-mapped SMILES for the full polymer chain.
        
    #     This is useful for force field parameterization.
        
    #     Returns:
    #         Atom-mapped SMILES string
    #     """
    #     mol = Chem.MolFromSmiles(self._generate_polymer_smiles())
    #     if mol is None:
    #         return ""
            
    #     # Add atom mapping
    #     for i, atom in enumerate(mol.GetAtoms()):
    #         atom.SetAtomMapNum(i + 1)
            
    #     return Chem.MolToSmiles(mol)


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