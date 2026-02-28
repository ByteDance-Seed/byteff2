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
        NOTE: This method only builds PDB structure files for Packmol packing.
        The actual GRO file (with correct atom count and H positions) is generated
        later by _generate_full_polymer_topology. So we do NOT need to embed the
        full DP=130 chain here — we just need any valid PDB with the right atoms.
        
        The PDB from this step will be REPLACED by _gro_to_pdb() in build_system(),
        which reads from the GRO file written by _generate_full_polymer_topology.
        So this step just needs to produce a placeholder.

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
                # Just save a placeholder PDB — the real coordinates come from
                # the GRO file generated by _generate_full_polymer_topology(),
                # which is converted to PDB via _gro_to_pdb() in build_system().
                # So we DON'T need to run the expensive _staged_embed for DP=130 here.
                output_path = os.path.join(output_dir, f"{name}.pdb")
                generated_structures[name] = output_path
                
                # Store full chain SMILES (built cheaply without 3D embedding)
                builder = PolymerChainBuilder(
                    monomer_smiles=polymer.monomer_smiles,
                    dp=polymer.degree_of_polymerization,
                    end_group_left=polymer.end_groups[0],
                    end_group_right=polymer.end_groups[1],
                    tacticity=polymer.tacticity,
                )
                polymer._full_chain_smiles = builder.get_polymer_smiles()
                
                logger.info(f"Registered polymer chain {name} with DP={polymer.degree_of_polymerization}")
                
        return generated_structures

    ## New change on 02-16-2026
    def build_system(self, total_atoms: int, components_ratio: dict, working_dir: str, 
                     build_gas: bool = False, reuse_if_exists: bool = True):
        """
        Override base Protocol.build_system() to use Packmol for polymer systems.
        
        Falls back to the base GROMACS method if no polymer components are present.
        """
        import shutil
        import subprocess
        import numpy as np
        from bytemol.toolkit.gmxtool.topparse import RecordAtomType, RecordMolecule, Records, TopoFullSystem
        
        # If no polymer components, use the base GROMACS method
        if not self.polymer_components:
            return super().build_system(total_atoms, components_ratio, working_dir, 
                                        build_gas, reuse_if_exists)
        
        logger.info(f'Building polymer system with Packmol for {list(components_ratio.keys())}')
        os.makedirs(working_dir, exist_ok=True)
        
        # --- Fast path: reuse existing packed system ---
        if reuse_if_exists:
            existing_gro = os.path.join(self.params_dir, 'solvent_salt.gro')
            existing_top = os.path.join(self.params_dir, 'system.top')
            if os.path.isfile(existing_gro) and os.path.isfile(existing_top):
                logger.info('Reusing existing system.top and solvent_salt.gro; skipping re-pack')
                from byteff2.toolkit.protocol import _parse_molecules_from_top
                mol_counts = _parse_molecules_from_top(existing_top)
                components = {}
                full_system_records, record_atomtype_names = [], []
                for component_name, count in mol_counts.items():
                    component = load_topo(self.params_dir, component_name)
                    component.molar_ratio = 1
                    component.molar_num = int(count)
                    components[component_name] = component
                    for record in component.atp_records.all:
                        if isinstance(record, RecordAtomType):
                            if record.name not in record_atomtype_names:
                                record_atomtype_names.append(record.name)
                                full_system_records.append(record)
                        else:
                            full_system_records.append(record)
                    full_system_records.extend(component.itp_records.all)
                self.config['natoms'] = int(sum(len(c.atoms) * c.molar_num for c in components.values()))
                return components
        
        # --- Load topology for each component ---
        components = {}
        full_system_records, record_atomtype_names = [], []
        system_charge = 0
        for component_name, molar_ratio in components_ratio.items():
            component = load_topo(self.params_dir, component_name)
            component.molar_ratio = molar_ratio
            components[component_name] = component
            for record in component.atp_records.all:
                if isinstance(record, RecordAtomType):
                    if record.name not in record_atomtype_names:
                        record_atomtype_names.append(record.name)
                        full_system_records.append(record)
                else:
                    full_system_records.append(record)
            system_charge += component.molar_ratio * component.net_charge
            full_system_records.extend(component.itp_records.all)
            # Copy param files to working dir
            for ext in ('.itp', '.atp', '.gro'):
                src = os.path.join(self.params_dir, f'{component_name}{ext}')
                dst = os.path.join(working_dir, f'{component_name}{ext}')
                if src != dst and os.path.isfile(src):
                    shutil.copy(src, dst)
        
        assert int(system_charge) == 0, f"System charge should be 0, but got {system_charge}"
        full_topparse = TopoFullSystem.from_records(full_system_records, sort_idx=False)
        
        # --- Determine molecule counts ---
        cfg = getattr(self, 'config', {}) if hasattr(self, 'config') else {}
        use_counts = False
        components_counts_from_cfg = None
        if isinstance(cfg, dict):
            if 'components_counts' in cfg and isinstance(cfg['components_counts'], dict):
                components_counts_from_cfg = cfg['components_counts']
                use_counts = True
            elif cfg.get('components_as_counts', False) or cfg.get('components_mode', '').lower() == 'counts':
                use_counts = True
        
        if use_counts:
            counts_source = components_counts_from_cfg or components_ratio
            full_topparse.molecules = []
            box_charge = 0
            for name, component in components.items():
                count = int(counts_source[name])
                component.molar_num = count
                full_topparse.molecules.append(
                    RecordMolecule.from_text(f"{component.name} {component.molar_num}")
                )
                box_charge += component.molar_num * component.net_charge
            real_total_atoms = int(sum(len(c.atoms) * c.molar_num for c in components.values()))
            try:
                self.config['natoms'] = real_total_atoms
            except Exception:
                pass
        else:
            input_mol_ratio = np.array(list(components_ratio.values()))
            from byteff2.toolkit.protocol import search_mixture
            real_total_atoms, mix = search_mixture(input_mol_ratio, total_atoms, total_atoms + 1000, components)
            full_topparse.molecules = []
            box_charge = 0
            for idx, component in enumerate(components.values()):
                component.molar_num = mix[idx]
                full_topparse.molecules.append(
                    RecordMolecule.from_text(f"{component.name} {component.molar_num}")
                )
                box_charge += component.molar_num * component.net_charge
        
        assert int(box_charge) == 0, f"Box charge should be 0, but got {box_charge}"
        
        # --- Compute box size ---
        init_density = predict_density(components)
        # Honor config overrides
        target_density = float(cfg.get('target_density', init_density)) if isinstance(cfg, dict) else init_density
        init_box = predict_box(components, target_density)
        
        if isinstance(cfg, dict):
            if cfg.get('box_length') is not None:
                box_nm = float(cfg['box_length'])
            elif cfg.get('box_scale') is not None:
                box_nm = float(init_box) * float(cfg['box_scale'])
            else:
                box_nm = init_box
        else:
            box_nm = init_box
        
        # Sort components by count (largest first) for consistent .top ordering
        components = {k: v for k, v in sorted(components.items(), key=lambda item: item[1].molar_num, reverse=True)}
        
        # --- Write system.top ---
        itp_list = [f'{mol_name}.itp' for mol_name in components.keys()]
        atp_list = [f'{mol_name}.atp' for mol_name in components.keys()]
        mols = [[i] for i in range(len(components))]
        with open(os.path.join(working_dir, 'system.top'), 'w') as f:
            f.write(full_topparse.strs_system_top_atp_itp(itp_list, atp_list, mols)[0])
        
        # --- Prepare PDB files for Packmol ---
        pdb_dir = os.path.join(working_dir, 'packmol_pdbs')
        os.makedirs(pdb_dir, exist_ok=True)
        
        component_pdbs = {}
        for comp_name, comp in components.items():
            pdb_path = os.path.join(pdb_dir, f'{comp_name}.pdb')

            if comp_name in self.polymer_components:
                # For polymers: generate PDB WITH hydrogens from the GRO file
                # (which has the correct atom count matching the topology)
                gro_path = os.path.join(self.params_dir, f'{comp_name}.gro')
                if os.path.isfile(gro_path):
                    self._gro_to_pdb(gro_path, pdb_path, comp_name)
                    logger.info(f"Converted polymer GRO to PDB for {comp_name} (with H)")
                else:
                    # Fallback: use the structure PDB but warn about H mismatch
                    struct_pdb = os.path.join(self.output_dir, 'structures', f'{comp_name}.pdb')
                    if os.path.isfile(struct_pdb):
                        shutil.copy(struct_pdb, pdb_path)
                        logger.warning(f"Using structure PDB for {comp_name} (may lack hydrogens!)")
                    else:
                        raise RuntimeError(f"No GRO or structure PDB found for polymer {comp_name}")
            else:
                # For small molecules (ions, solvents): generate PDB from GRO
                self._write_small_molecule_pdb(comp_name, pdb_path)
            
            # Verify PDB file is valid
            if not os.path.isfile(pdb_path) or os.path.getsize(pdb_path) < 10:
                raise RuntimeError(f"Failed to create valid PDB for {comp_name} at {pdb_path}")
            
            # Verify atom count matches topology
            pdb_atom_count = 0
            with open(pdb_path, 'r') as pf:
                for line in pf:
                    if line.startswith(('ATOM', 'HETATM')):
                        pdb_atom_count += 1
            topo_atom_count = len(comp.atoms)
            if pdb_atom_count != topo_atom_count:
                logger.warning(f"PDB atom count mismatch for {comp_name}: "
                             f"PDB has {pdb_atom_count}, topology has {topo_atom_count}")
            
            component_pdbs[comp_name] = pdb_path
        
        # --- Generate Packmol input ---
        box_ang = box_nm * 10.0  # nm -> Angstrom
        tolerance = 2.0
        margin = 2.0
        
        # Use ABSOLUTE paths for Packmol input to avoid cwd issues
        output_pdb = os.path.abspath(os.path.join(working_dir, 'system.pdb'))
        abs_component_pdbs = {name: os.path.abspath(p) for name, p in component_pdbs.items()}

        # --- Run Packmol with retries (expand box if needed) ---
        max_retries = 12
        for attempt in range(max_retries):
            logger.info(f"Packmol attempt {attempt + 1}/{max_retries}, box = {box_ang:.1f} Å ({box_ang/10:.3f} nm)")
            
            # Write Packmol input
            packmol_input_path = os.path.abspath(os.path.join(working_dir, 'packmol.inp'))
            with open(packmol_input_path, 'w') as f:
                f.write(f"tolerance {tolerance}\n")
                f.write("filetype pdb\n")
                f.write(f"output {output_pdb}\n")
                f.write("seed -1\n")
                # Add more iterations for difficult packing
                f.write("maxit 40\n")
                f.write("nloop 2000\n")
                f.write("\n")
                
                for comp_name, comp in components.items():
                    pdb_p = abs_component_pdbs[comp_name]
                    count = comp.molar_num
                    f.write(f"structure {pdb_p}\n")
                    f.write(f"  number {count}\n")
                    f.write(f"  inside box {margin} {margin} {margin} "
                            f"{box_ang - margin} {box_ang - margin} {box_ang - margin}\n")
                    f.write("end structure\n")
                    f.write("\n")
            
            # Log the packmol input for debugging
            with open(packmol_input_path, 'r') as f:
                logger.info(f"Packmol input:\n{f.read()}")
            
            try:
                # Run Packmol WITHOUT cwd override — paths are absolute
                with open(packmol_input_path, 'r') as fin:
                    result = subprocess.run(
                        ['packmol'],
                        stdin=fin,
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 min timeout for polymer systems
                    )
                
                # Log full output for debugging
                if result.stdout:
                    logger.info(f"Packmol stdout (last 20 lines):\n{chr(10).join(result.stdout.splitlines()[-20:])}")
                if result.stderr:
                    logger.warning(f"Packmol stderr:\n{result.stderr}")
                
                if result.returncode == 0 and os.path.isfile(output_pdb):
                    if os.path.getsize(output_pdb) > 100:
                        # Verify atom count
                        atom_count = 0
                        with open(output_pdb, 'r') as pf:
                            for line in pf:
                                if line.startswith(('ATOM', 'HETATM')):
                                    atom_count += 1
                        if atom_count >= real_total_atoms:
                            logger.info(f"Packmol completed successfully ({atom_count} atoms)")
                            break
                        else:
                            logger.warning(f"Packmol output has {atom_count} atoms, expected {real_total_atoms}")
                
                logger.warning(f"Packmol failed (code {result.returncode}); expanding box by 10%")
            except subprocess.TimeoutExpired:
                logger.warning("Packmol timed out; expanding box by 10%")
            except FileNotFoundError:
                raise RuntimeError("Packmol executable not found in PATH. "
                                   "Make sure packmol is installed and in your PATH.")
            
            box_ang *= 1.10
        else:
            raise RuntimeError(f"Packmol failed to pack system after {max_retries} retries")
        
        # --- Convert Packmol PDB to GRO ---
        box_nm_final = box_ang / 10.0
        output_gro = os.path.join(working_dir, 'solvent_salt.gro')
        
        gro_written = False
        for gmx_cmd in ['gmx', 'gmx_mpi']:
            try:
                result = subprocess.run(
                    [gmx_cmd, 'editconf', '-f', output_pdb, '-o', output_gro,
                     '-box', str(box_nm_final), str(box_nm_final), str(box_nm_final)],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0 and os.path.isfile(output_gro):
                    gro_written = True
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if not gro_written:
            # Manual PDB → GRO conversion as fallback
            logger.info("gmx editconf not available; converting PDB to GRO manually")
            self._pdb_to_gro(output_pdb, output_gro, box_nm_final)
        
        if not os.path.isfile(output_gro):
            raise RuntimeError(f"Failed to convert Packmol output to GRO")
        
        # --- Copy outputs to params_dir ---
        shutil.copy(os.path.join(working_dir, 'solvent_salt.gro'), 
                     os.path.join(self.params_dir, 'solvent_salt.gro'))
        shutil.copy(os.path.join(working_dir, 'system.top'), 
                     os.path.join(self.params_dir, 'system.top'))
        
        logger.info(f"Polymer system packed successfully: {real_total_atoms} atoms in {box_nm_final:.3f} nm box")
        return components

    ## New change on 02-19-2026
    def _write_small_molecule_pdb(self, mol_name: str, pdb_path: str):
        """
        Write a PDB file for a small molecule (ion/solvent) from its GRO file,
        using a direct format conversion that Packmol can reliably parse.
        
        Falls back to RDKit-based PDB generation from SMILES if GRO parsing fails.
        """
        gro_path = os.path.join(self.params_dir, f'{mol_name}.gro')
        
        # Try reading the GRO and writing a clean PDB manually
        if os.path.isfile(gro_path):
            try:
                self._gro_to_pdb(gro_path, pdb_path, mol_name)
                return
            except Exception as e:
                logger.warning(f"Failed to convert GRO to PDB for {mol_name}: {e}")
        
        # Fallback: generate from SMILES using RDKit
        smiles = None
        for comp in self._get_all_components():
            if comp.name == mol_name:
                smiles = comp.smiles
                break
        if smiles is None:
            smiles = self.config.get('smiles', {}).get(mol_name, '')
        
        if smiles:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                Chem.MolToPDBFile(mol, pdb_path)
                logger.info(f"Wrote PDB for {mol_name} from SMILES via RDKit")
                return
            except Exception as e:
                logger.warning(f"RDKit PDB generation failed for {mol_name}: {e}")
        
        raise RuntimeError(f"Could not generate PDB for {mol_name}")

    ## New change on 02-19-2026
    def _pdb_to_gro(self, pdb_path: str, gro_path: str, box_nm: float):
        """
        Convert a PDB file to GRO format manually.
        
        This is used as a fallback when gmx editconf is not available.
        """
        atoms = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    atomname = line[12:16].strip()
                    resname = line[17:20].strip()
                    resnr = int(line[22:26])
                    x = float(line[30:38]) / 10.0  # Angstrom to nm
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                    atoms.append((resnr, resname, atomname, x, y, z))
        
        with open(gro_path, 'w') as f:
            f.write(f"System packed by Packmol, converted by ByteFF2\n")
            f.write(f"{len(atoms):5d}\n")
            for i, (resnr, resname, atomname, x, y, z) in enumerate(atoms, 1):
                # Cap residue number at 99999
                rn = resnr % 100000
                f.write(f"{rn:5d}{resname:<5s}{atomname:>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            f.write(f"   {box_nm:.5f}   {box_nm:.5f}   {box_nm:.5f}\n")
        
        logger.info(f"Converted PDB to GRO: {len(atoms)} atoms, box={box_nm:.3f} nm")

    ## New change on 02-19-2026
    def _gro_to_pdb(self, gro_path: str, pdb_path: str, mol_name: str):
        """
        Convert a GRO file to PDB format for Packmol.
        
        Uses strict PDB fixed-width column format (HETATM records):
          Columns  1- 6: Record type "HETATM"
          Columns  7-11: Atom serial number (right-justified)
          Column  12:    Blank
          Columns 13-16: Atom name (see rules below)
          Column  17:    Alternate location indicator (blank)
          Columns 18-20: Residue name (right-justified)
          Column  21:    Blank
          Column  22:    Chain ID
          Columns 23-26: Residue sequence number (right-justified)
          Column  27:    Code for insertion of residues (blank)
          Columns 28-30: Blanks
          Columns 31-38: X coordinate (8.3f)
          Columns 39-46: Y coordinate (8.3f)
          Columns 47-54: Z coordinate (8.3f)
          Columns 55-60: Occupancy (6.2f)
          Columns 61-66: Temperature factor (6.2f)
          Columns 73-76: Segment ID (optional)
          Columns 77-78: Element symbol (right-justified)
        """
        atoms = []
        with open(gro_path, 'r') as f:
            lines = f.readlines()
        natoms = int(lines[1].strip())
        for i in range(2, 2 + natoms):
            line = lines[i]
            # GRO fixed-width: resid(5) resname(5) atomname(5) atomnr(5) x(8.3f) y(8.3f) z(8.3f)
            resname = line[5:10].strip()
            atomname = line[10:15].strip()
            x_nm = float(line[20:28])
            y_nm = float(line[28:36])
            z_nm = float(line[36:44])
            # Convert nm to Angstrom for PDB
            x_a = x_nm * 10.0
            y_a = y_nm * 10.0
            z_a = z_nm * 10.0
            atoms.append((atomname, resname, x_a, y_a, z_a))
        
        resname_pdb = (mol_name or 'MOL')[:3].upper()

        # Check for zero-coordinate atoms and fix them
        import random as _random
        _random.seed(42)
        n_fixed = 0
        for idx in range(len(atoms)):
            aname, rname, x, y, z = atoms[idx]
            if abs(x) < 1e-4 and abs(y) < 1e-4 and abs(z) < 1e-4:
                # Find a nearby non-zero atom to place this one near
                ref_x, ref_y, ref_z = 0.0, 0.0, 0.0
                # Look backwards for nearest non-zero atom
                for j in range(idx - 1, -1, -1):
                    _, _, rx, ry, rz = atoms[j]
                    if abs(rx) > 1e-4 or abs(ry) > 1e-4 or abs(rz) > 1e-4:
                        ref_x, ref_y, ref_z = rx, ry, rz
                        break
                # Place ~1 Å from reference in a random direction
                dx = _random.uniform(-1, 1)
                dy = _random.uniform(-1, 1)
                dz = _random.uniform(-1, 1)
                norm = max((dx*dx + dy*dy + dz*dz)**0.5, 1e-6)
                scale = 1.09 / norm
                atoms[idx] = (aname, rname, ref_x + dx * scale, ref_y + dy * scale, ref_z + dz * scale)
                n_fixed += 1
        
        if n_fixed > 0:
            logger.warning(f"Fixed {n_fixed} zero-coordinate atoms in GRO→PDB conversion for {mol_name}")
        
        with open(pdb_path, 'w') as f:
            f.write(f"REMARK   PDB for {mol_name} generated by ByteFF2 (from GRO)\n")
            for idx, (aname, rname, x, y, z) in enumerate(atoms, 1):
                # Determine element symbol from atom name (strip digits)
                elem_chars = ''.join(c for c in aname if c.isalpha())
                if elem_chars.upper().startswith('CL'):
                    elem = 'Cl'
                elif elem_chars.upper().startswith('BR'):
                    elem = 'Br'
                elif elem_chars.upper().startswith('LI'):
                    elem = 'Li'
                elif elem_chars.upper().startswith('NA'):
                    elem = 'Na'
                elif len(elem_chars) > 0:
                    elem = elem_chars[0].upper()
                else:
                    elem = 'X'
                
                # PDB atom name formatting rules (columns 13-16, 4 chars):
                # - 1-char element: starts in col 14, e.g. " C1 ", " H12"
                # - 2-char element: starts in col 13, e.g. "CL1 ", "BR  "
                # Truncate atom name to max 4 chars for PDB compliance
                aname_trunc = aname[:4]
                if len(elem) == 1:
                    # Right-justify element in col 13, then left-fill rest
                    atom_name_field = f" {aname_trunc:<3s}"
                else:
                    atom_name_field = f"{aname_trunc:<4s}"
                # Ensure exactly 4 chars
                atom_name_field = atom_name_field[:4]
                
                # Serial number: cap at 99999 (PDB limit for columns 7-11)
                serial = idx % 100000
                # Residue number: cap at 9999 (PDB limit for columns 23-26)
                resseq = 1
                
                # Build the line using strict column positions (1-indexed):
                # 1-6: record, 7-11: serial, 12: blank, 13-16: name,
                # 17: altloc, 18-20: resName, 21: blank, 22: chainID,
                # 23-26: resSeq, 27: iCode, 28-30: blanks,
                # 31-38: x, 39-46: y, 47-54: z, 55-60: occ, 61-66: bfac,
                # 77-78: element
                line = (
                    f"HETATM"                    # 1-6
                    f"{serial:5d}"               # 7-11
                    f" "                          # 12
                    f"{atom_name_field:4s}"       # 13-16
                    f" "                          # 17 (altloc)
                    f"{resname_pdb:>3s}"          # 18-20
                    f" "                          # 21
                    f" "                          # 22 (chain)
                    f"{resseq:4d}"               # 23-26
                    f" "                          # 27 (iCode)
                    f"   "                        # 28-30
                    f"{x:8.3f}"                  # 31-38
                    f"{y:8.3f}"                  # 39-46
                    f"{z:8.3f}"                  # 47-54
                    f"{1.0:6.2f}"                # 55-60 (occupancy)
                    f"{0.0:6.2f}"                # 61-66 (bfactor)
                    f"          "                 # 67-76 (padding)
                    f"{elem:>2s}"                 # 77-78
                    f"\n"
                )
                f.write(line)
            f.write("END\n")
        
        logger.info(f"Converted GRO to PDB for {mol_name}: {len(atoms)} atoms")

    ## New change on 02-13-2026
    def generate_ff_params_polymer(self, component_smiles: dict, force: bool = False):
        """
        Generate force field parameters for polymer systems.

        For polymers:
        1. Parameterize a representative oligomer (trimer) via the standard
           get_nb_params → tfs.write_itp() pipeline (identical to small molecules).
        2. Then replicate the trimer's topology records to build a full-chain
           ITP/ATP that is parser-compatible.
        3. Extrapolate the nonbonded params dict for OpenMM system build.

        For non-polymers: use the standard pipeline directly.
        """
        from rdkit import Chem
        from byteff2.train.utils import load_model
        from bytemol.utils import get_data_file_path
        from bytemol.core import Molecule

        model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
        model = load_model(os.path.dirname(model_dir))
        all_nb_params = {}

        for mol_name, smiles in component_smiles.items():
            logger.info(f'Preparing force field params for {mol_name}')

            is_polymer = mol_name in self.polymer_components

            # Step 1: Always parameterize via standard pipeline.
            # For polymers, `smiles` is already the representative oligomer SMILES.
            params = self._generate_small_molecule_params(model, mol_name, smiles, force)
            all_nb_params[mol_name] = params

            # Step 2: For polymers with DP > oligomer DP, replicate topology + params
            if is_polymer:
                polymer_comp = self.polymer_components[mol_name]
                full_dp = polymer_comp.degree_of_polymerization
                oligomer_dp = min(3, full_dp)
                if full_dp > oligomer_dp:
                    logger.info(f"Generating full polymer topology for {mol_name} (DP={full_dp})")
                    self._generate_full_polymer_topology(
                        mol_name, polymer_comp, params, force
                    )
                    # Also extrapolate the params dict for OpenMM
                    # Determine full chain atom count
                    from byteff2.toolkit.polymer_builder import PolymerChainBuilder
                    def _build_smiles(dp):
                        b = PolymerChainBuilder(
                            monomer_smiles=polymer_comp.monomer_smiles,
                            dp=dp,
                            end_group_left=polymer_comp.end_groups[0],
                            end_group_right=polymer_comp.end_groups[1],
                            tacticity=polymer_comp.tacticity,
                        )
                        return b.get_polymer_smiles()
                    
                    dimer_s = _build_smiles(2)
                    trimer_s = _build_smiles(oligomer_dp)
                    dn = Molecule.from_smiles(dimer_s, nconfs=0).natoms
                    tn = Molecule.from_smiles(trimer_s, nconfs=0).natoms
                    apr = tn - dn
                    ea = tn - oligomer_dp * apr
                    full_natoms = ea + full_dp * apr

                    full_params = self._extrapolate_nb_params_for_full_chain(
                        params, polymer_comp, full_natoms
                    )
                    all_nb_params[mol_name] = full_params
                    
                    # Cache full params
                    params_json_fp = os.path.join(self.params_dir, f'{mol_name}.json')
                    with open(params_json_fp, 'w') as f:
                        json.dump(full_params, f, indent=2)

        # Load metadata
        for mol_name in component_smiles:
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

    ## New change on 02-13-2026
    def _generate_small_molecule_params(self, model, mol_name: str, smiles: str, force: bool) -> dict:
        """
        Generate force field params for a small molecule (or oligomer) using the
        SAME pipeline as Protocol.generate_ff_params:
            Molecule.from_smiles → get_nb_params → tfs.write_itp(separated_atp=True)

        Returns the per-molecule params dict (charges, sigmas, epsilons, etc.).
        """
        from byteff2.train.utils import get_nb_params
        from byteff2.toolkit.protocol import write_gro
        from bytemol.core import Molecule

        itp_fp = os.path.join(self.params_dir, f'{mol_name}.itp')
        atp_fp = os.path.join(self.params_dir, f'{mol_name}.atp')
        gro_fp = os.path.join(self.params_dir, f'{mol_name}.gro')
        nb_meta_fp = os.path.join(self.params_dir, f'{mol_name}_nb_params.json')
        params_json_fp = os.path.join(self.params_dir, f'{mol_name}.json')

        have_all = all(os.path.isfile(p) for p in (itp_fp, atp_fp, gro_fp, params_json_fp))
        if have_all and not force:
            try:
                with open(params_json_fp) as fh:
                    params = json.load(fh)
                logger.info(f'Found cached params for {mol_name}; skipping regeneration')
                return params
            except Exception:
                logger.warning(f'Failed to load {params_json_fp}; will regenerate')

        logger.info(f'Generating force field params for {mol_name}')
        mol = Molecule.from_smiles(smiles, nconfs=1)
        mol.name = mol_name
        metadata, params, tfs, mol = get_nb_params(model, mol)

        # Use the standard writer — guaranteed to produce parser-compatible files
        tfs.write_itp(itp_fp, separated_atp=True)

        # Write GRO
        write_gro(mol, gro_fp)

        # Cache params and metadata
        with open(params_json_fp, 'w') as f:
            json.dump(params, f, indent=2)
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

    ## New change on 02-13-2026
    def _generate_full_polymer_topology(self, mol_name: str, polymer_comp,
                                         oligomer_params: dict, force: bool = False):
        """
        Generate full polymer topology (.itp, .atp, .gro) files for the FULL chain.

        Strategy
        --------
        1.  Load the trimer ITP/ATP that ``tfs.write_itp()`` already wrote
            (these are guaranteed parser-compatible).
        2.  Determine atoms-per-repeat-unit by comparing trimer vs dimer topologies
            generated through the SAME pipeline (Molecule.from_smiles → get_nb_params).
        3.  Identify repeating atom/bond/angle/dihedral patterns in the trimer topology.
        4.  Replicate interior repeat unit records to build full chain.
        5.  Handle atom type definitions (ATP) for the new atoms via TopoAtomTypes.
        6.  Write via ``TopoFullSystem.write_itp()`` for parser compatibility.
        """
        from byteff2.toolkit.polymer_builder import PolymerChainBuilder
        from byteff2.toolkit.protocol import write_gro
        from byteff2.train.utils import get_nb_params, load_model
        from bytemol.core import Molecule
        from bytemol.toolkit.gmxtool.topparse import (
            Records, TopoFullSystem, TopoMolecule, TopoAtomTypes,
            RecordAtomType, RecordAtom, RecordBond, RecordAngle,
            RecordDihedral, RecordPair, RecordExclusion,
            RecordMoleculeType, RecordSection,
        )
        from bytemol.utils import get_data_file_path
        import copy
        from rdkit import Chem
        from rdkit.Chem import AllChem

        itp_fp = os.path.join(self.params_dir, f'{mol_name}.itp')
        atp_fp = os.path.join(self.params_dir, f'{mol_name}.atp')
        gro_fp = os.path.join(self.params_dir, f'{mol_name}.gro')

        full_dp = polymer_comp.degree_of_polymerization
        oligomer_dp = min(3, full_dp)

        if full_dp <= oligomer_dp:
            return

        # =====================================================================
        # Step 1: Determine atoms_per_repeat using consistent pipeline
        # =====================================================================
        def _build_oligomer_smiles(dp):
            builder = PolymerChainBuilder(
                monomer_smiles=polymer_comp.monomer_smiles,
                dp=dp,
                end_group_left=polymer_comp.end_groups[0],
                end_group_right=polymer_comp.end_groups[1],
                tacticity=polymer_comp.tacticity,
            )
            return builder.get_polymer_smiles()

        dimer_smiles = _build_oligomer_smiles(2)
        trimer_smiles = _build_oligomer_smiles(oligomer_dp)

        dimer_mol_obj = Molecule.from_smiles(dimer_smiles, nconfs=0)
        trimer_mol_obj = Molecule.from_smiles(trimer_smiles, nconfs=0)
        dimer_natoms = dimer_mol_obj.natoms
        trimer_natoms = trimer_mol_obj.natoms

        atoms_per_repeat = trimer_natoms - dimer_natoms
        logger.info(f"Atoms per repeat unit: {atoms_per_repeat} "
                     f"(trimer={trimer_natoms}, dimer={dimer_natoms})")

        if atoms_per_repeat <= 0:
            raise ValueError(
                f"Cannot determine repeat unit size: trimer has {trimer_natoms} atoms, "
                f"dimer has {dimer_natoms} atoms"
            )

        monomer_smiles = _build_oligomer_smiles(1)
        monomer_mol_obj = Molecule.from_smiles(monomer_smiles, nconfs=0)
        monomer_natoms = monomer_mol_obj.natoms

        end_atoms = trimer_natoms - oligomer_dp * atoms_per_repeat
        logger.info(f"End group atoms: {end_atoms}, monomer_natoms: {monomer_natoms}")

        full_natoms_expected = end_atoms + full_dp * atoms_per_repeat
        logger.info(f"Expected full chain atoms: {full_natoms_expected}")

        # =====================================================================
        # Step 2: Load the trimer topology (ITP + ATP together)
        # =====================================================================
        # The trimer's write_itp(separated_atp=True) puts atom types in ATP
        # and molecule topology in ITP. We must load BOTH before from_records
        # so that add_record_atom can find the atom types.
        trimer_atp_records = Records.from_file(atp_fp, incdir=None, allow_unknown=False)
        trimer_itp_records = Records.from_file(itp_fp, incdir=None, allow_unknown=False)
        # Combine: ATP records first (so atomtypes are registered before atoms)
        combined_records = trimer_atp_records.all + trimer_itp_records.all
        trimer_tfs = TopoFullSystem.from_records(
            records=combined_records, sort_idx=False
        )

        trimer_mol_topo = trimer_tfs.mol_topos[0]

        topo_natoms = len(trimer_mol_topo.atoms)
        assert topo_natoms == trimer_natoms, (
            f"Trimer topology has {topo_natoms} atoms but Molecule has {trimer_natoms}"
        )

        # =====================================================================
        # Step 3: Compute section boundaries
        # =====================================================================
        left_cap_size = (end_atoms + 1) // 2
        right_cap_size = end_atoms - left_cap_size

        sec1_end = left_cap_size + 2 * atoms_per_repeat
        template_start = left_cap_size + atoms_per_repeat
        template_end = template_start + atoms_per_repeat
        sec3_start = template_end

        logger.info(f"Topology sections: sec1=[0:{sec1_end}], "
                     f"template=[{template_start}:{template_end}], "
                     f"sec3=[{sec3_start}:{trimer_natoms}]")

        extra_repeats = full_dp - oligomer_dp

        computed_full = sec1_end + extra_repeats * atoms_per_repeat + (trimer_natoms - sec3_start)
        logger.info(f"Computed full chain atoms: {computed_full}, expected: {full_natoms_expected}")

        sec3_offset = extra_repeats * atoms_per_repeat

        # =====================================================================
        # Step 4: Build full atom records
        # =====================================================================
        trimer_atoms = trimer_mol_topo.atoms
        trimer_type_indices = trimer_mol_topo.atom_to_type_index

        # Template atoms and their type indices
        template_atoms = trimer_atoms[template_start:template_end]
        template_type_indices = trimer_type_indices[template_start:template_end]

        full_atoms = []
        full_type_indices = []

        # Section 1: copy as-is
        for idx in range(sec1_end):
            atom = copy.deepcopy(trimer_atoms[idx])
            full_atoms.append(atom)
            full_type_indices.append(trimer_type_indices[idx])

        # Replicate template for extra_repeats
        for rep_idx in range(extra_repeats):
            for j, tmpl in enumerate(template_atoms):
                new_atom = copy.deepcopy(tmpl)
                new_nr = sec1_end + rep_idx * atoms_per_repeat + j + 1
                new_atom.nr = new_nr
                full_atoms.append(new_atom)
                full_type_indices.append(template_type_indices[j])

        # Section 3: shift indices
        for idx in range(sec3_start, trimer_natoms):
            new_atom = copy.deepcopy(trimer_atoms[idx])
            new_atom.nr = trimer_atoms[idx].nr + sec3_offset
            full_atoms.append(new_atom)
            full_type_indices.append(trimer_type_indices[idx])

        # Re-index all atoms sequentially
        for i, atom in enumerate(full_atoms):
            atom.nr = i + 1

        logger.info(f"Built {len(full_atoms)} atom records for full chain")

        # =====================================================================
        # Step 4b: Correct charges to ensure neutrality
        # =====================================================================
        # The replicated charges from the trimer don't sum exactly to 0.
        # We must fix them BEFORE writing the ITP, since the writer's
        # round_list_sum_to_int will fail if the gap exceeds allow_round_diff.
        total_charge = sum(a.charge for a in full_atoms)
        if abs(total_charge) > 1e-6:
            correction_per_atom = -total_charge / len(full_atoms)
            for a in full_atoms:
                a.charge += correction_per_atom
            new_total = sum(a.charge for a in full_atoms)
            logger.info(f"Charge neutrality correction on full atom records: "
                        f"{correction_per_atom:.8f}/atom (was {total_charge:.6f}, "
                        f"now {new_total:.10f})")
            
        # =====================================================================
        # Step 5: Build bonded interaction records
        # =====================================================================
        # Use .ai, .aj, .ak, .al (the actual attribute names in bytemol)

        def _classify_and_replicate(records, get_idx_fn, set_idx_fn):
            """Replicate bonded records from trimer to full chain."""
            result = []

            for rec in records:
                indices = get_idx_fn(rec)  # list of 1-based
                idx0 = [i - 1 for i in indices]  # 0-based

                all_in_sec1_only = all(i < template_start for i in idx0)
                all_in_sec3 = all(i >= sec3_start for i in idx0)
                all_in_sec1 = all(i < sec1_end for i in idx0)

                if all_in_sec1_only:
                    result.append(copy.deepcopy(rec))

                elif all_in_sec3:
                    new_rec = copy.deepcopy(rec)
                    set_idx_fn(new_rec, [i + sec3_offset for i in indices])
                    result.append(new_rec)

                elif all_in_sec1:
                    result.append(copy.deepcopy(rec))

                    all_in_template = all(template_start <= i < template_end for i in idx0)
                    if all_in_template:
                        for rep_idx in range(1, extra_repeats + 1):
                            shift = rep_idx * atoms_per_repeat
                            new_rec = copy.deepcopy(rec)
                            set_idx_fn(new_rec, [i + shift for i in indices])
                            result.append(new_rec)

                else:
                    # Bridges template (or sec1) and sec3
                    for rep_idx in range(extra_repeats + 1):
                        shift = rep_idx * atoms_per_repeat
                        new_rec = copy.deepcopy(rec)
                        new_indices = []
                        for i, i0 in zip(indices, idx0):
                            if i0 < sec3_start:
                                new_indices.append(i + shift)
                            else:
                                new_indices.append(i + sec3_offset - (extra_repeats - rep_idx) * atoms_per_repeat)
                        if all(1 <= ni <= len(full_atoms) for ni in new_indices):
                            set_idx_fn(new_rec, new_indices)
                            result.append(new_rec)

            # Deduplicate
            seen = set()
            deduped = []
            for rec in result:
                key = tuple(get_idx_fn(rec))
                if key not in seen:
                    seen.add(key)
                    deduped.append(rec)
            return deduped

        # Accessor functions using CORRECT attribute names: .ai, .aj, .ak, .al
        def get_bond_idx(r): return [r.ai, r.aj]
        def set_bond_idx(r, idx):
            r.ai, r.aj = idx[0], idx[1]

        def get_angle_idx(r): return [r.ai, r.aj, r.ak]
        def set_angle_idx(r, idx):
            r.ai, r.aj, r.ak = idx[0], idx[1], idx[2]

        def get_dihedral_idx(r): return [r.ai, r.aj, r.ak, r.al]
        def set_dihedral_idx(r, idx):
            r.ai, r.aj, r.ak, r.al = idx[0], idx[1], idx[2], idx[3]

        def get_pair_idx(r): return [r.ai, r.aj]
        def set_pair_idx(r, idx):
            r.ai, r.aj = idx[0], idx[1]

        trimer_bonds = trimer_mol_topo.bonds
        trimer_angles = trimer_mol_topo.angles
        trimer_dihedrals = trimer_mol_topo.dihedrals
        trimer_pairs = getattr(trimer_mol_topo, 'pairs', []) or []
        trimer_exclusions = getattr(trimer_mol_topo, 'exclusions', []) or []
        trimer_pairs_nb = getattr(trimer_mol_topo, 'pairs_nb', []) or []
        trimer_vsites1 = getattr(trimer_mol_topo, 'virtual_sites1', []) or []
        trimer_vsites2 = getattr(trimer_mol_topo, 'virtual_sites2', []) or []
        trimer_vsites3 = getattr(trimer_mol_topo, 'virtual_sites3', []) or []

        full_bonds = _classify_and_replicate(trimer_bonds, get_bond_idx, set_bond_idx)
        full_angles = _classify_and_replicate(trimer_angles, get_angle_idx, set_angle_idx)
        full_dihedrals = _classify_and_replicate(trimer_dihedrals, get_dihedral_idx, set_dihedral_idx)
        full_pairs = _classify_and_replicate(trimer_pairs, get_pair_idx, set_pair_idx)

        # Handle exclusions separately (different structure: .ai + .aj_list)
        full_exclusions = self._replicate_exclusions(
            trimer_exclusions, template_start, template_end,
            sec1_end, sec3_start, sec3_offset,
            extra_repeats, atoms_per_repeat, len(full_atoms)
        )

        # Handle virtual sites
        def get_vsite1_idx(r): return [r.av, r.ai]
        def set_vsite1_idx(r, idx): r.av, r.ai = idx[0], idx[1]

        def get_vsite2_idx(r): return [r.av, r.ai, r.aj]
        def set_vsite2_idx(r, idx): r.av, r.ai, r.aj = idx[0], idx[1], idx[2]

        def get_vsite3_idx(r): return [r.av, r.ai, r.aj, r.ak]
        def set_vsite3_idx(r, idx): r.av, r.ai, r.aj, r.ak = idx[0], idx[1], idx[2], idx[3]

        full_vsites1 = _classify_and_replicate(trimer_vsites1, get_vsite1_idx, set_vsite1_idx) if trimer_vsites1 else []
        full_vsites2 = _classify_and_replicate(trimer_vsites2, get_vsite2_idx, set_vsite2_idx) if trimer_vsites2 else []
        full_vsites3 = _classify_and_replicate(trimer_vsites3, get_vsite3_idx, set_vsite3_idx) if trimer_vsites3 else []

        logger.info(f"Bonded records: bonds={len(full_bonds)}, angles={len(full_angles)}, "
                     f"dihedrals={len(full_dihedrals)}, pairs={len(full_pairs)}, "
                     f"exclusions={len(full_exclusions)}")

        # =====================================================================
        # Step 6: Register new atom types in TopoAtomTypes
        # =====================================================================
        # The trimer has unique atom type names per atom. For new atoms
        # (replicated + shifted), we need new unique type names with the same
        # LJ parameters registered in TopoAtomTypes.

        ta = TopoAtomTypes(trimer_tfs.uuid)

        # For atoms in sec1 (indices 0..sec1_end-1), types are already registered.
        # For new atoms, create new type names and register them.
        for i in range(sec1_end, len(full_atoms)):
            atom = full_atoms[i]
            old_type = atom.atype
            old_type_idx = full_type_indices[i]
            old_atype_rec = ta.atomtypes[old_type_idx]

            new_type_name = f"{old_type}_{i+1}"
            if new_type_name in ta.type_to_index:
                # Already exists (shouldn't happen but be safe)
                full_type_indices[i] = ta.type_to_index[new_type_name]
            else:
                new_atype_rec = copy.deepcopy(old_atype_rec)
                new_atype_rec.name = new_type_name
                new_idx = ta.add_record(new_atype_rec)
                full_type_indices[i] = new_idx

            atom.atype = new_type_name

        # =====================================================================
        # Step 7: Reassemble TopoMolecule and write
        # =====================================================================
        full_mol_topo = trimer_tfs.mol_topos[0]

        # Replace all record lists
        full_mol_topo.atoms = full_atoms
        full_mol_topo.atom_to_type_index = full_type_indices
        full_mol_topo.bonds = full_bonds
        full_mol_topo.angles = full_angles
        full_mol_topo.dihedrals = full_dihedrals
        full_mol_topo.pairs = full_pairs
        full_mol_topo.exclusions = full_exclusions
        full_mol_topo.pairs_nb = _classify_and_replicate(trimer_pairs_nb, get_pair_idx, set_pair_idx) if trimer_pairs_nb else []
        full_mol_topo.virtual_sites1 = full_vsites1
        full_mol_topo.virtual_sites2 = full_vsites2
        full_mol_topo.virtual_sites3 = full_vsites3

        # Increase allow_round_diff to accommodate small residual after our correction,
        # or disable rounding entirely since charges are already corrected.
        full_mol_topo.allow_round_diff = 0.0  # skip rounding — charges are already correct
        
        # Write via the standard writer
        logger.info("Writing full polymer topology via TopoFullSystem.write_itp()")
        trimer_tfs.write_itp(itp_fp, idx=0, separated_atp=True)

        builder_full = PolymerChainBuilder(
            monomer_smiles=polymer_comp.monomer_smiles,
            dp=full_dp,
            end_group_left=polymer_comp.end_groups[0],
            end_group_right=polymer_comp.end_groups[1],
            tacticity=polymer_comp.tacticity,
        )
        # build_chain() now uses _staged_embed() internally,
        # producing a conformer with reasonable bond lengths
        full_rdkit = builder_full.build_chain()
        full_rdkit_h = Chem.AddHs(full_rdkit)
        
        # The builder already embedded the mol before RemoveHs;
        # re-add Hs and copy conformer coords for heavy atoms, then place Hs
        if full_rdkit_h.GetNumConformers() == 0 and full_rdkit.GetNumConformers() > 0:
            # Copy heavy atom coords from the embedded mol (without H)
            from rdkit.Geometry import Point3D
            conf_no_h = full_rdkit.GetConformer(0)
            new_conf = AllChem.Conformer(full_rdkit_h.GetNumAtoms())
            
            # Map heavy atoms: AddHs preserves heavy atom order
            heavy_idx = 0
            for i in range(full_rdkit_h.GetNumAtoms()):
                atom = full_rdkit_h.GetAtomWithIdx(i)
                if atom.GetAtomicNum() != 1:  # Not hydrogen
                    pos = conf_no_h.GetAtomPosition(heavy_idx)
                    new_conf.SetAtomPosition(i, pos)
                    heavy_idx += 1
                else:
                    # Temporarily place H at origin; will be fixed below
                    new_conf.SetAtomPosition(i, Point3D(0.0, 0.0, 0.0))
            
            full_rdkit_h.AddConformer(new_conf, assignId=True)
            
            # Now place H atoms at chemically reasonable positions
            # Method 1: Use AllChem.ConstrainedEmbed or partial optimization
            # Fix heavy atom positions and only optimize H positions
            logger.info(f"Placing H atoms for full polymer ({full_rdkit_h.GetNumAtoms()} atoms)")
            
            try:
                # Create a force field with heavy atoms constrained
                ff = AllChem.UFFGetMoleculeForceField(full_rdkit_h)
                if ff is not None:
                    # Constrain all heavy atoms (non-H) to their current positions
                    for i in range(full_rdkit_h.GetNumAtoms()):
                        if full_rdkit_h.GetAtomWithIdx(i).GetAtomicNum() != 1:
                            ff.AddFixedPoint(i)
                    
                    # Place H atoms near their bonded heavy atom before optimizing
                    conf = full_rdkit_h.GetConformer(0)
                    import random as _random
                    _random.seed(42)
                    for i in range(full_rdkit_h.GetNumAtoms()):
                        atom = full_rdkit_h.GetAtomWithIdx(i)
                        if atom.GetAtomicNum() == 1:
                            neighbors = atom.GetNeighbors()
                            if neighbors:
                                nb_pos = conf.GetAtomPosition(neighbors[0].GetIdx())
                                # Place ~1 Å from neighbor in a random direction
                                dx = _random.uniform(-1, 1)
                                dy = _random.uniform(-1, 1)
                                dz = _random.uniform(-1, 1)
                                norm = max((dx*dx + dy*dy + dz*dz)**0.5, 1e-6)
                                scale = 1.09 / norm  # typical C-H bond length
                                conf.SetAtomPosition(i, Point3D(
                                    nb_pos.x + dx * scale,
                                    nb_pos.y + dy * scale,
                                    nb_pos.z + dz * scale
                                ))
                    
                    # Now optimize only H positions (heavy atoms are fixed)
                    ff.Initialize()
                    converged = ff.Minimize(maxIts=1000, energyTol=1e-4, forceTol=1e-3)
                    logger.info(f"H-atom placement optimization: converged={converged}, "
                               f"energy={ff.CalcEnergy():.1f}")
                else:
                    raise RuntimeError("UFF force field creation failed")

            except Exception as e:
                logger.warning(f"Force field H placement failed: {e}; using geometric placement")
                # Fallback: place H atoms geometrically near their bonded neighbor
                conf = full_rdkit_h.GetConformer(0)
                import random as _random
                _random.seed(42)
                for i in range(full_rdkit_h.GetNumAtoms()):
                    atom = full_rdkit_h.GetAtomWithIdx(i)
                    if atom.GetAtomicNum() == 1:
                        pos = conf.GetAtomPosition(i)
                        if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6:
                            neighbors = atom.GetNeighbors()
                            if neighbors:
                                nb_pos = conf.GetAtomPosition(neighbors[0].GetIdx())
                                dx = _random.uniform(-1, 1)
                                dy = _random.uniform(-1, 1)
                                dz = _random.uniform(-1, 1)
                                norm = max((dx*dx + dy*dy + dz*dz)**0.5, 1e-6)
                                scale = 1.09 / norm
                                conf.SetAtomPosition(i, Point3D(
                                    nb_pos.x + dx * scale,
                                    nb_pos.y + dy * scale,
                                    nb_pos.z + dz * scale
                                ))
                            else:
                                # Isolated H — shouldn't happen but handle gracefully
                                conf.SetAtomPosition(i, Point3D(
                                    _random.uniform(-2, 2),
                                    _random.uniform(-2, 2),
                                    _random.uniform(-2, 2)
                                ))
            
            # Final verification: ensure no atoms are at exactly (0,0,0)
            conf = full_rdkit_h.GetConformer(0)
            n_zero = 0
            for i in range(full_rdkit_h.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6:
                    n_zero += 1
            if n_zero > 0:
                logger.error(f"WARNING: {n_zero} atoms still at origin after H placement!")
            else:
                logger.info(f"All {full_rdkit_h.GetNumAtoms()} atoms have non-zero coordinates")
        
        elif full_rdkit_h.GetNumConformers() > 0:
            # Conformer exists but check for zero-coordinate H atoms
            conf = full_rdkit_h.GetConformer(0)
            from rdkit.Geometry import Point3D
            import random as _random
            _random.seed(42)
            n_fixed = 0
            for i in range(full_rdkit_h.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6:
                    atom = full_rdkit_h.GetAtomWithIdx(i)
                    neighbors = atom.GetNeighbors()
                    if neighbors:
                        nb_pos = conf.GetAtomPosition(neighbors[0].GetIdx())
                        dx = _random.uniform(-1, 1)
                        dy = _random.uniform(-1, 1)
                        dz = _random.uniform(-1, 1)
                        norm = max((dx*dx + dy*dy + dz*dz)**0.5, 1e-6)
                        scale = 1.09 / norm
                        conf.SetAtomPosition(i, Point3D(
                            nb_pos.x + dx * scale,
                            nb_pos.y + dy * scale,
                            nb_pos.z + dz * scale
                        ))
                    n_fixed += 1
            if n_fixed > 0:
                logger.info(f"Fixed {n_fixed} zero-coordinate atoms in existing conformer")
                # Optionally optimize H positions with constrained heavy atoms
                try:
                    ff = AllChem.UFFGetMoleculeForceField(full_rdkit_h)
                    if ff is not None:
                        for i in range(full_rdkit_h.GetNumAtoms()):
                            if full_rdkit_h.GetAtomWithIdx(i).GetAtomicNum() != 1:
                                ff.AddFixedPoint(i)
                        ff.Initialize()
                        ff.Minimize(maxIts=500)
                        logger.info("Optimized H positions with constrained heavy atoms")
                except Exception as e:
                    logger.warning(f"H optimization failed (non-fatal): {e}")

        # Write GRO directly from RDKit conformer (bypassing Molecule)
        self._write_gro_from_rdkit(full_rdkit_h, gro_fp, mol_name)

    def _write_gro_from_rdkit(self, rdkit_mol, gro_path: str, mol_name: str):
        """
        Write a GRO file directly from an RDKit molecule with explicit Hs and 3D coords.
        
        This bypasses the bytemol Molecule class which may not handle large polymers
        with explicit H correctly.
        """
        from rdkit import Chem
        
        if rdkit_mol.GetNumConformers() == 0:
            raise RuntimeError(f"RDKit mol for {mol_name} has no conformer")
        
        conf = rdkit_mol.GetConformer(0)
        natoms = rdkit_mol.GetNumAtoms()
        resname = (mol_name or 'MOL')[:5]
        
        lines = []
        lines.append(f"GRO file created by ByteFF2 for {mol_name}\n")
        lines.append(f"{natoms:5d}\n")
        
        for i in range(natoms):
            atom = rdkit_mol.GetAtomWithIdx(i)
            elem = atom.GetSymbol()
            atomnm = f"{elem}{i+1}"[:5]
            pos = conf.GetAtomPosition(i)
            # Convert Angstrom to nm
            x_nm = pos.x / 10.0
            y_nm = pos.y / 10.0
            z_nm = pos.z / 10.0
            # GRO format: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            lines.append(
                f"{1:5d}{resname:<5s}{atomnm:>5s}{(i+1) % 100000:5d}"
                f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}\n"
            )
        
        # Minimal box; will be replaced by editconf later
        lines.append("   1.00000   1.00000   1.00000\n")
        
        with open(gro_path, 'w') as f:
            f.writelines(lines)
        
        logger.info(f"Wrote GRO from RDKit mol for {mol_name}: {natoms} atoms")

    def _replicate_exclusions(self, trimer_exclusions, template_start, template_end,
                               sec1_end, sec3_start, sec3_offset,
                               extra_repeats, atoms_per_repeat, total_atoms):
        """Replicate exclusion records from trimer to full chain.
        
        RecordExclusion has .ai (int) and .aj_list (list[int]), all 1-based.
        """
        import copy

        result = []
        for excl in trimer_exclusions:
            ai = excl.ai       # 1-based
            ai0 = ai - 1       # 0-based
            aj_list = excl.aj_list  # list of 1-based

            if ai0 < template_start:
                # In left_cap + repeat_0: keep as-is
                result.append(copy.deepcopy(excl))
                # If ai is in template region (repeat_1), replicate
                if ai0 >= template_start:
                    pass  # handled below
            elif ai0 < sec1_end:
                # In template region (repeat_1)
                result.append(copy.deepcopy(excl))
                # Also replicate for extra_repeats
                if template_start <= ai0 < template_end:
                    for rep_idx in range(1, extra_repeats + 1):
                        shift = rep_idx * atoms_per_repeat
                        new_excl = copy.deepcopy(excl)
                        new_excl.ai = ai + shift
                        new_excl.aj_list = []
                        for aj in aj_list:
                            aj0 = aj - 1
                            if aj0 < sec3_start:
                                new_aj = aj + shift
                            else:
                                new_aj = aj + sec3_offset - (extra_repeats - rep_idx) * atoms_per_repeat
                            if 1 <= new_aj <= total_atoms:
                                new_excl.aj_list.append(new_aj)
                        new_excl.aj_list.sort()
                        if new_excl.aj_list:
                            result.append(new_excl)
            elif ai0 >= sec3_start:
                # In section 3: shift by sec3_offset
                new_excl = copy.deepcopy(excl)
                new_excl.ai = ai + sec3_offset
                new_excl.aj_list = [aj + sec3_offset for aj in aj_list
                                     if 1 <= aj + sec3_offset <= total_atoms]
                new_excl.aj_list.sort()
                if new_excl.aj_list:
                    result.append(new_excl)
            else:
                # Bridges template→sec3: replicate
                for rep_idx in range(extra_repeats + 1):
                    shift = rep_idx * atoms_per_repeat
                    new_excl = copy.deepcopy(excl)
                    new_excl.ai = ai + shift
                    new_excl.aj_list = []
                    for aj in aj_list:
                        aj0 = aj - 1
                        if aj0 < sec3_start:
                            new_aj = aj + shift
                        else:
                            new_aj = aj + sec3_offset - (extra_repeats - rep_idx) * atoms_per_repeat
                        if 1 <= new_aj <= total_atoms:
                            new_excl.aj_list.append(new_aj)
                    new_excl.aj_list.sort()
                    if new_excl.aj_list:
                        result.append(new_excl)

        return result

    def _extrapolate_nb_params_for_full_chain(self, oligomer_params: dict,
                                               polymer_comp, full_natoms: int) -> dict:
        """
        Extrapolate nonbonded parameters from oligomer to full chain.
        
        Uses the same repeat-unit logic as topology replication:
        atoms in the interior get parameters from the template (middle repeat unit).
        
        Args:
            oligomer_params: Parameter dict from trimer (from get_nb_params)
            polymer_comp: Polymer component info
            full_natoms: Expected number of atoms in the full chain
            
        Returns:
            Full-chain parameter dict with the same keys as oligomer_params
        """
        from byteff2.toolkit.polymer_builder import PolymerChainBuilder
        from bytemol.core import Molecule
        
        oligomer_dp = min(3, polymer_comp.degree_of_polymerization)
        full_dp = polymer_comp.degree_of_polymerization

        def _build_smiles(dp):
            b = PolymerChainBuilder(
                monomer_smiles=polymer_comp.monomer_smiles,
                dp=dp,
                end_group_left=polymer_comp.end_groups[0],
                end_group_right=polymer_comp.end_groups[1],
                tacticity=polymer_comp.tacticity,
            )
            return b.get_polymer_smiles()

        dimer_smiles = _build_smiles(2)
        trimer_smiles = _build_smiles(oligomer_dp)
        
        dimer_natoms = Molecule.from_smiles(dimer_smiles, nconfs=0).natoms
        trimer_natoms = Molecule.from_smiles(trimer_smiles, nconfs=0).natoms
        atoms_per_repeat = trimer_natoms - dimer_natoms
        end_atoms = trimer_natoms - oligomer_dp * atoms_per_repeat
        left_cap_size = (end_atoms + 1) // 2

        template_start = left_cap_size + atoms_per_repeat
        template_end = template_start + atoms_per_repeat
        sec3_start = template_end
        sec1_end = left_cap_size + 2 * atoms_per_repeat
        extra_repeats = full_dp - oligomer_dp

        # Build full params by replicating the pattern
        full_params = {}
        for key in oligomer_params:
            val = oligomer_params[key]
            if not isinstance(val, list):
                full_params[key] = val
                continue
            
            if len(val) != trimer_natoms:
                # This key doesn't have per-atom values (or wrong length) — copy as-is
                full_params[key] = val
                continue

            # Section 1: left_cap + repeat_0 + repeat_1 (first sec1_end atoms)
            new_vals = list(val[:sec1_end])
            
            # Template values (repeat_1 = middle repeat unit)
            template_vals = val[template_start:template_end]
            
            # Replicate template for extra_repeats
            for _ in range(extra_repeats):
                new_vals.extend(template_vals)
            
            # Section 3: repeat_2 + right_cap
            new_vals.extend(val[sec3_start:])

            full_params[key] = new_vals

        # Verify lengths
        for key, val in full_params.items():
            if isinstance(val, list) and len(val) == full_natoms:
                pass  # correct
            elif isinstance(val, list) and len(val) != full_natoms and len(oligomer_params.get(key, [])) == trimer_natoms:
                logger.warning(f"Param '{key}' has {len(val)} values, expected {full_natoms}. "
                               f"Truncating/padding.")
                if len(val) > full_natoms:
                    full_params[key] = val[:full_natoms]
                else:
                    # Pad with last template value
                    pad_val = template_vals[-1] if template_vals else 0.0
                    full_params[key] = val + [pad_val] * (full_natoms - len(val))

        # Ensure charge neutrality for the polymer molecule itself
        # (The polymer should be neutral; ions are handled separately)
        charge_key = None
        for candidate in ('charge', 'charges'):
            if candidate in full_params and isinstance(full_params[candidate], list):
                charge_key = candidate
                break
        
        if charge_key and len(full_params[charge_key]) == full_natoms:
            total_q = sum(full_params[charge_key])
            if abs(total_q) > 0.01:
                n = len(full_params[charge_key])
                correction = -total_q / n
                full_params[charge_key] = [c + correction for c in full_params[charge_key]]
                logger.info(f"Charge neutrality correction: {correction:.6f}/atom "
                            f"(total was {total_q:.4f})")

        return full_params

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
            f.write("; ai    aj    funct    c0(nm)    c1(kJ/mol/nm^2)\n")
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx() + 1
                j = bond.GetEndAtomIdx() + 1
                f.write(f"  {i:5d}  {j:5d}  1\n")
                # must be 5 fields for parser compatibility
                # f.write(f"  {i:5d}  {j:5d}  1  0.1540  265000.0\n")
            
            # Write angles (basic angle detection)
            f.write("\n[ angles ]\n")
            f.write("; ai    aj    ak    funct    c0(deg)    c1(kJ/mol/rad^2)\n")
            
            for atom in mol.GetAtoms():
                j = atom.GetIdx()
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(neighbors) >= 2:
                    for idx1 in range(len(neighbors)):
                        for idx2 in range(idx1 + 1, len(neighbors)):
                            i = neighbors[idx1] + 1
                            k = neighbors[idx2] + 1
                            f.write(f"  {i:5d}  {j+1:5d}  {k:5d}  1\n")
                            # 6 fields for parser compatibility
                            # f.write(f"  {i:5d}  {j+1:5d}  {k:5d}  1  109.50  520.0\n")
            
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

    ## New change on 02-13-2026
    def _create_molecule_from_rdkit(self, rdkit_mol, name: str):
        """
        Create a bytemol Molecule from an RDKit mol with explicit Hs and 3D coords.
        
        This is needed because Molecule.from_smiles() may produce a different
        atom ordering or H count than our builder.
        """
        from bytemol.core import Molecule, Conformer
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np

        # Ensure 3D coordinates
        if rdkit_mol.GetNumConformers() == 0:
            try:
                params = AllChem.ETKDGv3()
                params.useRandomCoords = True
                params.maxIterations = 5000
                AllChem.EmbedMolecule(rdkit_mol, params)
            except Exception:
                AllChem.EmbedMolecule(rdkit_mol, useRandomCoords=True)

        conf = rdkit_mol.GetConformer(0)
        natoms = rdkit_mol.GetNumAtoms()
        
        coords = np.zeros((natoms, 3))
        symbols = []
        for i in range(natoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
            symbols.append(rdkit_mol.GetAtomWithIdx(i).GetSymbol())

        # Build Molecule via mapped SMILES if possible, else via Conformer
        try:
            # Try the canonical path: generate mapped SMILES
            # Add atom map numbers
            for atom in rdkit_mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            mapped_smi = Chem.MolToSmiles(rdkit_mol)
            # Clear map nums
            for atom in rdkit_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            
            mol = Molecule.from_mapped_smiles(mapped_smi, nconfs=0)
            mol.name = name
            # Add conformer
            from bytemol.core import Conformer as BConformer
            bconf = BConformer(coords=coords, symbols=symbols)
            mol._conformers = [bconf]
            return mol
        except Exception as e:
            logger.warning(f"Could not create Molecule from mapped SMILES: {e}")
            # Fallback: create from SMILES without Hs, then add conformer
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(rdkit_mol))
                mol = Molecule.from_smiles(smi, nconfs=1)
                mol.name = name
                return mol
            except Exception as e2:
                logger.warning(f"Fallback Molecule creation also failed: {e2}")
                # Last resort: create a minimal Molecule
                from bytemol.core import Conformer as BConformer
                bconf = BConformer(coords=coords, symbols=symbols)
                # Use the Molecule(xyz_path) constructor pattern
                # by writing a temporary XYZ
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w', delete=False) as f:
                    f.write(f"{natoms}\n")
                    f.write(f"name={name}\n")
                    for sym, (x, y, z) in zip(symbols, coords):
                        f.write(f"{sym}  {x:.6f}  {y:.6f}  {z:.6f}\n")
                    tmp_path = f.name
                mol = Molecule(tmp_path, name=name)
                os.unlink(tmp_path)
                return mol

    def _staged_minimization(self, top, system, positions, unit_cell):
        """
        Perform staged energy minimization for polymer systems.
        
        Packmol-packed polymer systems often have severe steric clashes that
        cause the standard L-BFGS minimizer to stall for hours (especially
        with polarizable Amoeba force field on CPU).
        
        Strategy:
        1. First minimize with only bonded forces (groups 0), capping iterations
        2. Then minimize with all forces, but with a hard iteration cap
        3. If energy is still very high, run a few steps of low-temperature
           Langevin dynamics to shake out remaining clashes
        
        Args:
            top: OpenMM Topology
            system: OpenMM System  
            positions: Initial positions (with units)
            unit_cell: Box dimensions
            
        Returns:
            Relaxed positions (with units)
        """
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        import time as _time
        
        logger.info("Starting staged pre-minimization for polymer system")
        
        # Select platform
        platform = None
        platform_properties = {}
        plat_name = os.environ.get('BYTEFF2_OPENMM_PLATFORM', '')
        prec = os.environ.get('BYTEFF2_OPENMM_PRECISION', 'mixed')
        
        if plat_name:
            try:
                platform = mm.Platform.getPlatformByName(plat_name)
                if plat_name in ('CUDA', 'OpenCL'):
                    platform_properties['Precision'] = prec
                logger.info(f"Using platform: {plat_name} ({prec})")
            except Exception:
                logger.warning(f"Requested platform '{plat_name}' not available; using default")
                platform = None
        
        if platform is None:
            # Auto-select best available: CUDA > OpenCL > CPU > Reference
            for pname in ['CUDA', 'OpenCL', 'CPU']:
                try:
                    platform = mm.Platform.getPlatformByName(pname)
                    if pname in ('CUDA', 'OpenCL'):
                        platform_properties['Precision'] = 'mixed'
                    logger.info(f"Auto-selected platform: {pname}")
                    break
                except Exception:
                    continue
        
        # =====================================================================
        # Stage 1: Minimization with BONDED forces only (very fast)
        # =====================================================================
        # Create a copy of the system with only bonded forces active.
        # This resolves the worst bond/angle distortions without the expensive
        # polarizable multipole evaluation.
        logger.info("Stage 1: Bonded-only minimization (max 1000 iterations)")
        t0 = _time.time()
        
        bonded_system = _copy.deepcopy(system)
        # Identify force groups: disable nonbonded (group 1) forces
        # by setting their force group and then excluding them
        for force in bonded_system.getForces():
            fname = force.__class__.__name__
            if fname in ('AmoebaMultipoleForce', 'CustomNonbondedForce',
                         'NonbondedForce', 'AmoebaVdwForce'):
                # Set a very high force group so we can exclude it
                force.setForceGroup(31)  # won't be evaluated in minimization
        
        integrator1 = mm.VerletIntegrator(0.001 * unit.picoseconds)
        if platform:
            sim1 = app.Simulation(top, bonded_system, integrator1, platform, platform_properties)
        else:
            sim1 = app.Simulation(top, bonded_system, integrator1)
        
        sim1.context.setPositions(positions)
        if unit_cell is not None:
            sim1.context.setPeriodicBoxVectors(
                mm.Vec3(unit_cell[0].value_in_unit(unit.nanometer), 0, 0) * unit.nanometer,
                mm.Vec3(0, unit_cell[1].value_in_unit(unit.nanometer), 0) * unit.nanometer,
                mm.Vec3(0, 0, unit_cell[2].value_in_unit(unit.nanometer)) * unit.nanometer,
            )
        
        state = sim1.context.getState(getEnergy=True)
        e0 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        logger.info(f"  Initial bonded energy: {e0:.1f} kJ/mol")
        
        try:
            # tolerance is a FORCE tolerance: kJ/mol/nm
            sim1.minimizeEnergy(maxIterations=1000,
                                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer)
        except Exception as e:
            logger.warning(f"  Bonded minimization exception: {e}")
        
        state = sim1.context.getState(getEnergy=True, getPositions=True)
        e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        positions = state.getPositions()
        dt1 = _time.time() - t0
        logger.info(f"  After bonded minimization: {e1:.1f} kJ/mol ({dt1:.1f} s)")
        
        del sim1, integrator1, bonded_system
        
        # =====================================================================
        # Stage 2: Full-system minimization with iteration cap
        # =====================================================================
        logger.info("Stage 2: Full-system minimization (max 500 iterations)")
        t1 = _time.time()
        
        integrator2 = mm.LangevinMiddleIntegrator(
            10 * unit.kelvin,
            1.0 / unit.picosecond,
            0.5 * unit.femtoseconds,
        )
        
        if platform:
            sim2 = app.Simulation(top, system, integrator2, platform, platform_properties)
        else:
            sim2 = app.Simulation(top, system, integrator2)
        
        sim2.context.setPositions(positions)
        if unit_cell is not None:
            sim2.context.setPeriodicBoxVectors(
                mm.Vec3(unit_cell[0].value_in_unit(unit.nanometer), 0, 0) * unit.nanometer,
                mm.Vec3(0, unit_cell[1].value_in_unit(unit.nanometer), 0) * unit.nanometer,
                mm.Vec3(0, 0, unit_cell[2].value_in_unit(unit.nanometer)) * unit.nanometer,
            )
        
        state = sim2.context.getState(getEnergy=True)
        e_full_init = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        logger.info(f"  Full-system initial energy: {e_full_init:.1f} kJ/mol")
        
        try:
            sim2.minimizeEnergy(maxIterations=500,
                                tolerance=100.0 * unit.kilojoules_per_mole / unit.nanometer)
        except Exception as e:
            logger.warning(f"  Full minimization exception: {e}")
        
        state = sim2.context.getState(getEnergy=True, getPositions=True)
        e2 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        positions = state.getPositions()
        dt2 = _time.time() - t1
        logger.info(f"  After full minimization: {e2:.1f} kJ/mol ({dt2:.1f} s)")
        
        # Check for NaN
        import math
        if math.isnan(e2):
            logger.error("Energy is NaN after Stage 2 minimization!")
            # Don't try Langevin — it will just produce more NaN
            # Fall back to bonded-only positions
            logger.warning("Falling back to bonded-only minimized positions")
            del sim2, integrator2
            return positions
        
        # =====================================================================
        # Stage 3: Low-temperature Langevin dynamics (only if energy is finite)
        # =====================================================================
        if e2 > 0 or abs(e2) > 1e8:
            logger.info("Stage 3: Low-temperature Langevin dynamics (10 K, 0.5 fs)")
            t2 = _time.time()
            
            try:
                sim2.context.setVelocitiesToTemperature(10 * unit.kelvin)
                
                n_batches = 10
                steps_per_batch = 100
                for batch in range(n_batches):
                    try:
                        sim2.step(steps_per_batch)
                    except Exception as e:
                        logger.warning(f"  Langevin batch {batch} failed: {e}")
                        # Try minimizing again before continuing
                        try:
                            sim2.minimizeEnergy(
                                maxIterations=100,
                                tolerance=1000.0 * unit.kilojoules_per_mole / unit.nanometer
                            )
                        except Exception:
                            pass
                        break
                    
                    # Check for NaN after each batch
                    state = sim2.context.getState(getEnergy=True)
                    e_check = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    if math.isnan(e_check):
                        logger.warning(f"  NaN detected at Langevin batch {batch}; stopping")
                        # Revert to pre-Langevin positions
                        sim2.context.setPositions(positions)
                        break
                
                state = sim2.context.getState(getEnergy=True, getPositions=True)
                e3 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                if not math.isnan(e3):
                    positions = state.getPositions()
                dt3 = _time.time() - t2
                logger.info(f"  After Langevin: {e3:.1f} kJ/mol ({dt3:.1f} s)")
            except Exception as e:
                logger.warning(f"  Stage 3 failed (non-fatal): {e}")
        
        # =====================================================================
        # Stage 4: Final minimization
        # =====================================================================
        logger.info("Stage 4: Final minimization (max 2000 iterations)")
        t3 = _time.time()
        
        sim2.context.setPositions(positions)
        try:
            sim2.minimizeEnergy(maxIterations=2000,
                                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer)
        except Exception as e:
            logger.warning(f"  Final minimization exception: {e}")
        
        state = sim2.context.getState(getEnergy=True, getPositions=True)
        e_final = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        if not math.isnan(e_final):
            positions = state.getPositions()
        dt4 = _time.time() - t3
        
        total_time = _time.time() - t0
        logger.info(f"  Final energy: {e_final:.1f} kJ/mol ({dt4:.1f} s)")
        logger.info(f"Staged pre-minimization complete in {total_time:.1f} s")
        
        del sim2, integrator2
        
        return positions

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
        
        # --- Platform selection ---
        if isinstance(self.config, dict):
            plat = self.config.get('openmm_platform')
            prec = self.config.get('openmm_precision')
            if plat:
                os.environ['BYTEFF2_OPENMM_PLATFORM'] = str(plat)
            if prec:
                os.environ['BYTEFF2_OPENMM_PRECISION'] = str(prec)

        # --- Pre-minimization for polymer systems ---
        input_positions = self._staged_minimization(
            input_top, input_system, input_positions, unit_cell
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
        
        # --- Platform selection ---
        # Force OpenCL or CUDA if available (critical for polarizable FF performance)
        if isinstance(self.config, dict):
            plat = self.config.get('openmm_platform')
            prec = self.config.get('openmm_precision')
            if plat:
                os.environ['BYTEFF2_OPENMM_PLATFORM'] = str(plat)
            if prec:
                os.environ['BYTEFF2_OPENMM_PRECISION'] = str(prec)

        # --- Pre-minimization: staged soft-core relaxation for polymer systems ---
        # Packmol-packed polymer systems often have significant steric clashes.
        # Running a staged pre-minimization prevents the main minimizer from
        # stalling on enormous forces for hours.
        input_positions = self._staged_minimization(
            input_top, input_system, input_positions, unit_cell
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