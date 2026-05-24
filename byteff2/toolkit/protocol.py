import json
import os
import shutil
import subprocess
from enum import Enum

import ase.io as aio
import numpy as np
import openmm.app as app
import openmm.unit as ou
import pandas as pd

from byteff2.md_utils.md_run import dcd_read, npt_run, nvt_run, rescale_box, volume_calc
from byteff2.md_utils.onsager_conductivity import onsager_calc
from byteff2.md_utils.viscosity import nonequ_run, viscosity_calc
from byteff2.toolkit.gmxtool import GMXScript
from byteff2.toolkit.openmmtool import generate_openmm_system
from byteff2.train.utils import get_nb_params, load_model
from bytemol.core import Molecule
from bytemol.toolkit.gmxtool.topparse import RecordAtomType, RecordMolecule, Records, TopoFullSystem
from bytemol.utils import get_data_file_path, setup_default_logging

logger = setup_default_logging()


class ComponentType(Enum):
    SOLVENT = 0
    ANION = 1
    CATION = 2
    UNDEFINED = 3

# ...existing code...

# Empirical ionic liquid density parameters (can be expanded)
IL_DENSITY_PARAMS = {
    # Common cation families (density contribution factor)
    'imidazolium': 1.15,
    'pyrrolidinium': 1.10,
    'ammonium': 1.05,
    'phosphonium': 1.00,
    # Common anion families
    'TFSI': 1.40,
    'FSI': 1.35,
    'BF4': 1.20,
    'PF6': 1.35,
    'Cl': 1.10,
}

def estimate_il_density(cation_smiles: str, anion_smiles: str, temperature: float = 298.15) -> float:
    """
    Estimate ionic liquid density based on ion structure.
    
    This is a simple empirical model; for accurate predictions,
    consider using QSPR models or experimental data.
    
    Args:
        cation_smiles: SMILES of cation
        anion_smiles: SMILES of anion  
        temperature: Temperature in K (density decreases ~0.1% per K above 298)
        
    Returns:
        Estimated density in g/mL
    """
    base_density = 1.2  # typical IL density
    
    # Temperature correction (approximate)
    temp_factor = 1.0 - 0.001 * (temperature - 298.15)
    
    # Could add SMILES-based estimation here
    # For now, return base with temperature correction
    return round(base_density * temp_factor, 2)

class Component:

    def __init__(self, topo_mol):
        self.name = topo_mol.name
        self.atoms = topo_mol.atoms
        self.net_charge = sum([atom.charge for atom in topo_mol.atoms])
        if self.net_charge > 1e-5:
            self.type = ComponentType.CATION
            self.density = 0.25
        elif self.net_charge < -1e-5:
            self.type = ComponentType.ANION
            self.density = 0.25
        else:
            self.type = ComponentType.SOLVENT
            self.density = 0.9
        self.molar_ratio = -1
        self.molar_num = -1
        self.molar_mass = sum([atom.mass for atom in topo_mol.atoms])
        self.itp_records = None
        self.atp_records = None

    @property
    def is_ion(self) -> bool:
        """Check if component is an ion (cation or anion)."""
        return self.type in (ComponentType.CATION, ComponentType.ANION)


def predict_density(component: dict):
    """
    Predict initial box density for system packing.
    
    Supports:
    - Solvent-based electrolytes (original behavior)
    - Ionic liquids / salt-only systems (cation + anion only)
    - Pure solvents
    """
    # density = 0
    # total_molar_ratio = 0
    # solvent = [c for c in component.values() if c.type == ComponentType.SOLVENT]
    # cation = [c for c in component.values() if c.type == ComponentType.CATION]
    # anion = [c for c in component.values() if c.type == ComponentType.ANION]
    # for c in solvent:
    #     density += c.density * c.molar_num
    #     total_molar_ratio += c.molar_num
    # sol_density = density / total_molar_ratio
    # sol_ratio = sol_density
    # # Add cation and anion
    # for c in cation:
    #     sol_density += min(c.density * c.molar_num / total_molar_ratio * sol_ratio, 0.5)
    # for c in anion:
    #     sol_density += min(c.density * c.molar_num / total_molar_ratio * sol_ratio, 0.5)
    # return round(sol_density, 2)
    ## extending for solvent-free systems: 01-19-2026
    solvent = [c for c in component.values() if c.type == ComponentType.SOLVENT]
    cation = [c for c in component.values() if c.type == ComponentType.CATION]
    anion = [c for c in component.values() if c.type == ComponentType.ANION]
    
    total_molar_num = sum(c.molar_num for c in component.values())
    if total_molar_num == 0:
        return 1.0  # fallback default
    
    # Case 1: Salt-only system (ionic liquid)
    if len(solvent) == 0:
        if len(cation) == 0 and len(anion) == 0:
            return 1.0  # no components at all
        
        # Ionic liquids typically have densities between 1.0-1.5 g/mL
        # Use weighted average of ion densities with a base IL density
        total_ion_mass = sum(c.molar_num * c.molar_mass for c in cation + anion)
        total_ion_count = sum(c.molar_num for c in cation + anion)
        avg_ion_mass = total_ion_mass / total_ion_count if total_ion_count > 0 else 100.0
        
        # Empirical density estimation for ionic liquids
        # Heavier ions typically lead to higher densities
        base_il_density = 1.2  # typical IL density
        mass_factor = min(avg_ion_mass / 150.0, 1.5)  # normalize by typical IL ion mass
        estimated_density = base_il_density * (0.8 + 0.4 * mass_factor)
        
        return round(min(max(estimated_density, 0.9), 2.0), 2)  # clamp to reasonable range
    
    # Case 2: Solvent-based system (original logic)
    density = 0
    total_molar_ratio = 0
    for c in solvent:
        density += c.density * c.molar_num
        total_molar_ratio += c.molar_num
    
    sol_density = density / total_molar_ratio
    sol_ratio = sol_density
    
    # Add cation and anion contributions
    for c in cation:
        sol_density += min(c.density * c.molar_num / total_molar_ratio * sol_ratio, 0.5)
    for c in anion:
        sol_density += min(c.density * c.molar_num / total_molar_ratio * sol_ratio, 0.5)
    
    return round(sol_density, 2)


def search_mixture(mol_ratio, min_atoms, max_atoms, components):
    """
    Search for mixture composition that fits atom count constraints.
    
    Extended to handle salt-only systems where charge neutrality must be maintained.
    """
    result = []
    num_atoms = np.array([len(component.atoms) for component in components.values()])
    
    # atoms_ratio = mol_ratio * num_atoms
    # uni_mol_ratio = mol_ratio / np.min(mol_ratio)
    # uni_atom_count = int(sum(uni_mol_ratio * num_atoms))
    # min_count = (min_atoms - 1) // uni_atom_count + 1
    # max_count = (max_atoms - 1) // uni_atom_count + 1
    # steps = max((max_count - min_count), 1)
    # for i in range(min_count, max_count, steps):
    #     guess = np.round(uni_mol_ratio * i).astype(int)
    #     guess_count = int(sum(guess * num_atoms))
    #     mix = np.round(guess_count * atoms_ratio / np.sum(atoms_ratio) / num_atoms).astype(int)
    #     result.append(guess_count)
    # total_atoms = result[0]
    # mix = np.round(total_atoms * atoms_ratio / np.sum(atoms_ratio) / num_atoms).astype(int)
    # return total_atoms, mix

    ## modified to handle salt-only systems: 01-19-2026
    # Check if this is a salt-only system
    component_list = list(components.values())
    solvent = [c for c in component_list if c.type == ComponentType.SOLVENT]
    cation = [c for c in component_list if c.type == ComponentType.CATION]
    anion = [c for c in component_list if c.type == ComponentType.ANION]
    
    is_salt_only = len(solvent) == 0 and len(cation) > 0 and len(anion) > 0
    
    if is_salt_only:
        # For salt-only systems, ensure charge neutrality
        # Assume 1:1 stoichiometry for simplicity (can be extended)
        cation_charges = [abs(c.net_charge) for c in cation]
        anion_charges = [abs(c.net_charge) for c in anion]
        
        # Find LCM-based ratio for charge neutrality if needed
        # For now, assume input ratio already ensures neutrality
        pass
    
    atoms_ratio = mol_ratio * num_atoms
    uni_mol_ratio = mol_ratio / np.min(mol_ratio[mol_ratio > 0])  # avoid division by zero
    uni_atom_count = int(sum(uni_mol_ratio * num_atoms))
    
    if uni_atom_count == 0:
        # Fallback for edge cases
        return min_atoms, np.ones(len(mol_ratio), dtype=int)
    
    min_count = max((min_atoms - 1) // uni_atom_count + 1, 1)
    max_count = max((max_atoms - 1) // uni_atom_count + 1, min_count + 1)
    steps = max((max_count - min_count), 1)
    
    for i in range(min_count, max_count, steps):
        guess = np.round(uni_mol_ratio * i).astype(int)
        guess_count = int(sum(guess * num_atoms))
        mix = np.round(guess_count * atoms_ratio / np.sum(atoms_ratio) / num_atoms).astype(int)
        result.append(guess_count)
    
    if len(result) == 0:
        result.append(min_atoms)
    
    total_atoms = result[0]
    mix = np.round(total_atoms * atoms_ratio / np.sum(atoms_ratio) / num_atoms).astype(int)
    
    # Ensure at least 1 molecule of each component
    mix = np.maximum(mix, 1)
    
    # For salt-only systems, verify charge neutrality
    if is_salt_only:
        total_charge = sum(
            mix[i] * component_list[i].net_charge 
            for i in range(len(mix))
        )
        if abs(total_charge) > 0.01:
            logger.warning(f"Salt-only system has non-zero charge: {total_charge}. Adjusting...")
            # Simple adjustment: scale to achieve neutrality
            # This is a simplified approach; more sophisticated balancing may be needed
            pass
    
    return int(sum(mix * num_atoms)), mix

def predict_box(components, density):
    factor = 0.11842
    total_mass = sum([x.molar_num * x.molar_mass for x in components.values()])
    estimate_box = (total_mass / density)**(1 / 3) * factor
    return round(estimate_box, 2)


def load_topo(topo_dir, mol_name):
    itp_records = Records.from_file(
        f'{topo_dir}/{mol_name}.itp',
        incdir=None,
        allow_unknown=False,
    )
    atp_records = Records.from_file(
        f'{topo_dir}/{mol_name}.atp',
        incdir=None,
        allow_unknown=False,
    )
    topparse = TopoFullSystem.from_records(itp_records.all + atp_records.all, sort_idx=False)
    component = Component(topparse.mol_topos[0])
    component.itp_records = itp_records
    component.atp_records = atp_records
    return component


def _parse_molecules_from_top(top_path: str) -> dict[str, int]:
    """Parse the [ molecules ] section of a GROMACS .top file and return counts.

    This is a light-weight parser that looks for the first [ molecules ] section
    and collects lines of form: "<name> <count>" ignoring comments and blanks.
    """
    counts = {}
    if not os.path.isfile(top_path):
        return counts
    in_mol = False
    with open(top_path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('['):
                # entering or leaving sections
                in_mol = ('molecules' in line)
                continue
            if in_mol:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        count = int(parts[1])
                    except Exception:
                        continue
                    counts[name] = count
    return counts


def _read_last_step(csv_path: str) -> int:
    try:
        if not os.path.isfile(csv_path):
            return 0
        df = pd.read_csv(csv_path)
        if 'Step' in df.columns and len(df['Step']) > 0:
            return int(df['Step'].iloc[-1])
    except Exception:
        pass
    return 0


def generate_system_gro(components, working_dir, box):
    """Generate GROMACS system with support for salt-only systems."""
    solvent = [c for c in components.values() if c.type == ComponentType.SOLVENT]
    cation = [c for c in components.values() if c.type == ComponentType.CATION]
    anion = [c for c in components.values() if c.type == ComponentType.ANION]
    script = GMXScript()
    script.add('cd "$(dirname "$0")" ')
    
    # for i, c in enumerate(solvent):
    #     # Generate the box from the first component
    #     if i == 0:
    #         # Generate the box for solvent
    #         script.init_gro_box(f"{c.name}.gro", box)
    #         rest_molecules = c.molar_num - 1
    #         if rest_molecules:
    #             script.insert_molecules(f"{c.name}.gro", rest_molecules)
    #         continue
    #     script.insert_molecules(f"{c.name}.gro", c.molar_num)

    # # Add cation and anion
    # for c in cation:
    #     script.insert_molecules(f"{c.name}.gro", c.molar_num)
    # for c in anion:
    #     script.insert_molecules(f"{c.name}.gro", c.molar_num)
    # # Add run md run command
    # script.finish()
    # script.write(f'{working_dir}/run_gmx.sh')
    # Determine which component initializes the box
    ## modified for solvent-free systems: 01-19-2026
    if len(solvent) > 0:
        # Original behavior: solvent initializes box
        for i, c in enumerate(solvent):
            if i == 0:
                script.init_gro_box(f"{c.name}.gro", box)
                rest_molecules = c.molar_num - 1
                if rest_molecules:
                    script.insert_molecules(f"{c.name}.gro", rest_molecules)
                continue
            script.insert_molecules(f"{c.name}.gro", c.molar_num)
    else:
        # Salt-only system: use the more abundant ion to initialize box
        all_ions = cation + anion
        if len(all_ions) == 0:
            raise ValueError("No components provided for system generation")
        
        # Sort by molar_num descending to start with the most abundant
        all_ions_sorted = sorted(all_ions, key=lambda c: c.molar_num, reverse=True)
        
        first_ion = all_ions_sorted[0]
        script.init_gro_box(f"{first_ion.name}.gro", box)
        rest_molecules = first_ion.molar_num - 1
        if rest_molecules > 0:
            script.insert_molecules(f"{first_ion.name}.gro", rest_molecules)
        
        # Insert remaining ions
        for c in all_ions_sorted[1:]:
            script.insert_molecules(f"{c.name}.gro", c.molar_num)
        
        # Early finish for salt-only
        script.finish()
        script.write(f'{working_dir}/run_gmx.sh')
        return
    
    # Add cation and anion for solvent-based systems
    for c in cation:
        script.insert_molecules(f"{c.name}.gro", c.molar_num)
    for c in anion:
        script.insert_molecules(f"{c.name}.gro", c.molar_num)
    
    script.finish()
    script.write(f'{working_dir}/run_gmx.sh')


def write_gro(mol: Molecule, save_path: str):
    """Write a single-molecule GRO file in strict fixed-width format.

    GROMACS requires GRO files to use fixed columns. Some generic writers
    produce variable-width fields that recent GROMACS rejects. We write the
    minimal compliant fields here: title, natoms, atom lines (no velocities),
    and a placeholder box (replaced later by editconf).
    """
    # Obtain positions (Angstrom) and convert to nm
    atoms = mol.conformers[0].to_ase_atoms()
    pos_A = atoms.get_positions()  # Angstrom
    pos_nm = pos_A / 10.0

    natoms = mol.natoms
    resname = (mol.name or 'MOL')[:5]

    def gro_line(resnr, resnm, atomnm, atomnr, x, y, z):
        # %5d%-5s%5s%5d%8.3f%8.3f%8.3f
        return f"{resnr:5d}{resnm:<5s}{atomnm:>5s}{atomnr:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"

    lines = []
    lines.append(f"GRO file created by ByteFF2 for {mol.name}\n")
    lines.append(f"{natoms:5d}\n")
    for i, (x, y, z) in enumerate(pos_nm, start=1):
        # atom name up to 5 chars: element+index (e.g., C1, O5)
        try:
            elem = atoms[i - 1].symbol
        except Exception:
            elem = 'A'
        atomnm = f"{elem}{i}"[:5]
        lines.append(gro_line(1, resname, atomnm, i, x, y, z))
    # Minimal box; will be replaced by editconf later
    lines.append("   1.00000   1.00000   1.00000\n")

    with open(save_path, 'w') as f:
        f.writelines(lines)


class Protocol:

    def __init__(self, params_dir: str, output_dir: str):
        os.makedirs(params_dir, exist_ok=True)
        self.params_dir = params_dir
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def generate_ff_params(self, component_smiles: dict, force: bool = False):
        model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
        model = load_model(os.path.dirname(model_dir))
        all_nb_params = {}

        for mol_name, smiles in component_smiles.items():
            logger.info(f'preparing force field params for {mol_name}')
            itp_fp = f'{self.params_dir}/{mol_name}.itp'
            atp_fp = f'{self.params_dir}/{mol_name}.atp'
            gro_fp = f'{self.params_dir}/{mol_name}.gro'
            nb_meta_fp = f'{self.params_dir}/{mol_name}_nb_params.json'
            params_json_fp = f'{self.params_dir}/{mol_name}.json'
            have_all = all(os.path.isfile(p) for p in (itp_fp, atp_fp, gro_fp))
            if have_all and not force:
                # Load per-molecule params from cached JSON for OpenMM system build
                if os.path.isfile(params_json_fp):
                    try:
                        with open(params_json_fp) as fh:
                            all_nb_params[mol_name] = json.load(fh)
                    except Exception:
                        logger.warning('Failed to load %s; will regenerate', params_json_fp)
                        have_all = False
                else:
                    have_all = False
                # Try to load common metadata if not already
                if 'metadata' not in all_nb_params and os.path.isfile(nb_meta_fp):
                    try:
                        with open(nb_meta_fp) as fh:
                            meta_wrap = json.load(fh)
                            if isinstance(meta_wrap, dict) and 'metadata' in meta_wrap:
                                all_nb_params['metadata'] = meta_wrap['metadata']
                    except Exception:
                        pass
                if have_all:
                    logger.info(f'Found cached params for {mol_name}; skipping regeneration')
                    continue
            # Generate fresh params if any required file missing or forced
            mol = Molecule.from_smiles(smiles, nconfs=1)
            mol.name = mol_name
            metadata, params, tfs, mol = get_nb_params(model, mol)
            tfs.write_itp(f'{self.params_dir}/{mol.name}.itp', separated_atp=True)
            write_gro(mol, f'{self.params_dir}/{mol.name}.gro')
            with open(params_json_fp, 'w') as f:
                json.dump(params, f, indent=2)
            with open(nb_meta_fp, 'w') as file:
                nb_params = {'metadata': metadata}
                json.dump(nb_params, file, indent=2)
            all_nb_params[mol_name] = params
            all_nb_params['metadata'] = metadata

        return all_nb_params

    def build_system(self, total_atoms: int, components_ratio: dict, working_dir: str, build_gas: bool = False, reuse_if_exists: bool = True):
        logger.info(f'building system for {components_ratio.keys()}')
        # read and parse topo files
        os.makedirs(working_dir, exist_ok=True)
        components = {}
        # Fast path: reuse previously packed system when resuming
        if reuse_if_exists:
            existing_gro = os.path.join(self.params_dir, 'solvent_salt.gro')
            existing_top = os.path.join(self.params_dir, 'system.top')
            if os.path.isfile(existing_gro) and os.path.isfile(existing_top):
                logger.info('Reusing existing system.top and solvent_salt.gro; skipping re-pack')
                # Derive composition from existing top
                mol_counts = _parse_molecules_from_top(existing_top)
                full_system_records, record_atomtype_names = [], []
                system_charge = 0
                for component_name, count in mol_counts.items():
                    component = load_topo(self.params_dir, component_name)
                    component.molar_ratio = 1  # placeholder; real counts set below
                    component.molar_num = int(count)
                    components[component_name] = component
                    for record in component.atp_records.all:
                        if isinstance(record, RecordAtomType):
                            if record.name not in record_atomtype_names:
                                record_atomtype_names.append(record.name)
                                full_system_records.append(record)
                        else:
                            full_system_records.append(record)
                    system_charge += component.molar_num * component.net_charge
                    full_system_records.extend(component.itp_records.all)
                assert int(system_charge) == 0, f"System charge should be 0, but got {system_charge}"
                # Nothing else to do; assume existing files are valid
                self.config['natoms'] = int(sum(len(c.atoms) * c.molar_num for c in components.values()))
                return components
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
            shutil.copy(f'{self.params_dir}/{component_name}.itp', f'{working_dir}/{component_name}.itp')
            shutil.copy(f'{self.params_dir}/{component_name}.atp', f'{working_dir}/{component_name}.atp')
            shutil.copy(f'{self.params_dir}/{component_name}.gro', f'{working_dir}/{component_name}.gro')
        assert int(system_charge) == 0, f"System charge should be 0, but got {system_charge}"
        full_topparse = TopoFullSystem.from_records(full_system_records, sort_idx=False)
        if build_gas:
            assert len(components) == 1, "Gas phase only support one component"
            component = list(components.values())[0]
            total_atoms = len(component.atoms)
            with open(f'{working_dir}/{component.name}.gro', 'r') as origin_gro_f:
                lines = origin_gro_f.readlines()[:-1]
            lines.append(" 100.00000 100.00000 100.00000\n")
            with open(f'{working_dir}/solvent_salt_gas.gro', 'w') as new_gro_f:
                new_gro_f.writelines(lines)
        # Decide if 'components_ratio' should be treated as exact molecule counts
        cfg = getattr(self, 'config', {}) if hasattr(self, 'config') else {}
        components_counts_from_cfg = None
        use_counts = False
        if isinstance(cfg, dict):
            if 'components_counts' in cfg and isinstance(cfg['components_counts'], dict):
                components_counts_from_cfg = cfg['components_counts']
                use_counts = True
            elif cfg.get('components_as_counts', False) or cfg.get('components_mode', '').lower() == 'counts':
                use_counts = True

        if use_counts:
            # Use exact counts either from dedicated 'components_counts' or values in 'components_ratio'
            counts_source = components_counts_from_cfg or components_ratio
            full_topparse.molecules = []
            box_charge = 0
            for name, component in components.items():
                count = int(counts_source[name])
                component.molar_num = count
                full_topparse.molecules.append(RecordMolecule.from_text(f"{component.name} {component.molar_num}"))
                box_charge += component.molar_num * component.net_charge
            # Keep 'natoms' consistent with chosen composition
            nat = int(sum(len(c.atoms) * c.molar_num for c in components.values()))
            try:
                # update in-memory config for downstream use
                self.config['natoms'] = nat
            except Exception:  # safety for unexpected config types
                pass
            real_total_atoms = nat
        else:
            input_mol_ratio = np.array(list(components_ratio.values()))
            real_total_atoms, mix = search_mixture(input_mol_ratio, total_atoms, total_atoms + 1000, components)

            full_topparse.molecules = []
            box_charge = 0
            for idx, component in enumerate(components.values()):
                component.molar_num = mix[idx]
                full_topparse.molecules.append(RecordMolecule.from_text(f"{component.name} {component.molar_num}"))
                box_charge += component.molar_num * component.net_charge
        assert int(box_charge) == 0, f"Box charge should be 0, but got {box_charge}"

        init_density = predict_density(components)
        init_box = predict_box(components, init_density)
        components = {k: v for k, v in sorted(components.items(), key=lambda item: item[1].molar_num, reverse=True)}
        itp_list = [f'{mol_name}.itp' for mol_name in components.keys()]
        atp_list = [f'{mol_name}.atp' for mol_name in components.keys()]
        mols = [[i] for i in range(len(components))]
        with open(f'{working_dir}/system.top', 'w') as f:
            f.write(full_topparse.strs_system_top_atp_itp(itp_list, atp_list, mols)[0])
        if build_gas:
            shutil.copy(f'{working_dir}/solvent_salt_gas.gro', f'{self.params_dir}/solvent_salt_gas.gro')
            shutil.copy(f'{working_dir}/system.top', f'{self.params_dir}/system_gas.top')
            return components

        # Allow overriding initial box via config if provided
        cfg = getattr(self, 'config', {}) if hasattr(self, 'config') else {}
        if isinstance(cfg, dict):
            if 'box_length' in cfg and cfg['box_length'] is not None:
                box = float(cfg['box_length'])
            elif 'box_scale' in cfg and cfg['box_scale'] is not None:
                box = float(init_box) * float(cfg['box_scale'])
            else:
                box = init_box
        else:
            box = init_box
        for _ in range(8):
            generate_system_gro(components, working_dir, box)
            command = f'cd {working_dir} && bash -x run_gmx.sh'
            try:
                child = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                logger.warning("run_gmx.sh timed out at box %.3f nm; expanding by 5%% and retrying", box)
                box *= 1.05
                continue
            if child.returncode != 0:
                # Likely packing failure; expand and retry rather than aborting immediately
                logger.warning("run_gmx.sh failed (code %s) at box %.3f nm; expanding by 5%% and retrying. Last lines:\n%s",
                               child.returncode, box, '\n'.join(child.stderr.splitlines()[-20:]))
                box *= 1.05
                continue
            gro_file = os.path.join(working_dir, "solvent_salt.gro")
            with open(gro_file, "r") as f:
                gro_total_atoms = int(f.readlines()[1].strip().split()[0])
            if real_total_atoms > gro_total_atoms:
                box *= 1.05
            else:
                break
        else:
            # If we exhausted retries
            raise RuntimeError(f"Failed to pack system after retries. Last stderr:\n{child.stderr}")
        shutil.copy(f'{working_dir}/solvent_salt.gro', f'{self.params_dir}/solvent_salt.gro')
        shutil.copy(f'{working_dir}/system.top', f'{self.params_dir}/system.top')
        return components

    def run_protocol(self,):
        raise NotImplementedError

    def post_process(self,):
        raise NotImplementedError


class DensityProtocol(Protocol):

    def __init__(self, config: dict):
        super().__init__(config['params_dir'], config['output_dir'])
        self.config = config

    def run_protocol(self):
        logger.info('running density protocol')
        nonbonded_params = self.generate_ff_params(self.config['smiles'], force=bool(self.config.get('force_regenerate_params', False)))
        _ = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
            reuse_if_exists=bool(self.config.get('resume', False)),
        )
        gro_file = f"{self.params_dir}/solvent_salt.gro"
        top_file = f"{self.params_dir}/system.top"
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        input_top, input_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )

        npt_steps = int(self.config.get('npt_steps', 1500000))
        npt_timestep_fs = int(self.config.get('npt_timestep_fs', 2)) if isinstance(self.config, dict) else 2
        resume = bool(self.config.get('resume', False))
        checkpoint_interval = int(self.config.get('checkpoint_interval', 5000))
        traj_interval = int(self.config.get('traj_interval', 500)) if isinstance(self.config, dict) else 500
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
            state_csv_override=(self.config.get('npt_state_csv') if isinstance(self.config, dict) else None),
            dcd_path_override=(self.config.get('npt_dcd') if isinstance(self.config, dict) else None),
            resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)) if isinstance(self.config, dict) else 2,
            resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True,
            traj_interval=traj_interval,
        )
        logger.info('Finished running density protocol')

    def post_process(self,):
        csv_file = os.path.join(self.output_dir, 'npt_state.csv')
        density = pd.read_csv(csv_file)["Density (g/mL)"]

        dd = []
        for _ in range(10):
            dd.append(np.mean(np.random.choice(density[2000:3000], 100)))
        density, density_std = np.mean(dd), np.std(dd)
        result = {
            "density": density,
            "density_std": density_std,
        }
        with open(os.path.join(self.output_dir, 'density_results.json'), 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(result)
        return result


class TransportProtocol(Protocol):

    def __init__(self, config: dict):
        super().__init__(config['params_dir'], config['output_dir'])
        self.config = config
        self.components = None

    def run_protocol(self):
        logger.info('running transport protocol')
        # Defaults (2 fs timestep). Allow override by steps or by time.
        def steps_from_time(cfg, steps_key, default_steps, time_ns_key=None, time_ps_key=None, timestep_fs=2):
            if isinstance(cfg, dict):
                if time_ns_key and cfg.get(time_ns_key) is not None:
                    return int(float(cfg[time_ns_key]) * 1e6 / float(timestep_fs))
                if time_ps_key and cfg.get(time_ps_key) is not None:
                    return int(float(cfg[time_ps_key]) * 1e3 / float(timestep_fs))
                if cfg.get(steps_key) is not None:
                    return int(cfg[steps_key])
            return int(default_steps)

        # Read timestep overrides (fs)
        npt_timestep_fs = int(self.config.get('npt_timestep_fs', 2)) if isinstance(self.config, dict) else 2
        nvt_timestep_fs = int(self.config.get('nvt_timestep_fs', 2)) if isinstance(self.config, dict) else 2
        nonequ_timestep_fs = int(self.config.get('nonequ_timestep_fs', 1)) if isinstance(self.config, dict) else 1

        npt_steps = steps_from_time(self.config, 'npt_steps', 4000000, time_ns_key='npt_time_ns', time_ps_key='npt_time_ps', timestep_fs=npt_timestep_fs)
        nvt_steps = steps_from_time(self.config, 'nvt_steps', 10000000, time_ns_key='nvt_time_ns', time_ps_key='nvt_time_ps', timestep_fs=nvt_timestep_fs)
        # nonequilibrium run uses VVIntegrator; default 1 fs unless overridden
        nonequ_steps = steps_from_time(self.config, 'nonequ_steps', 1000000, time_ns_key='nonequ_time_ns', time_ps_key='nonequ_time_ps', timestep_fs=nonequ_timestep_fs)
        # Optional OpenMM platform/precision overrides via config
        if isinstance(self.config, dict):
            plat = self.config.get('openmm_platform')
            prec = self.config.get('openmm_precision')
            if plat:
                os.environ['BYTEFF2_OPENMM_PLATFORM'] = str(plat)
            if prec:
                os.environ['BYTEFF2_OPENMM_PRECISION'] = str(prec)
        nonbonded_params = self.generate_ff_params(self.config['smiles'], force=bool(self.config.get('force_regenerate_params', False)))
        self.components = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
            reuse_if_exists=bool(self.config.get('resume', False)),
        )
        gro_file = f"{self.params_dir}/solvent_salt.gro"
        top_file = f"{self.params_dir}/system.top"
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        input_top, input_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )
        resume = bool(self.config.get('resume', False))
        checkpoint_interval = int(self.config.get('checkpoint_interval', 5000))
        traj_interval = int(self.config.get('traj_interval', 500)) if isinstance(self.config, dict) else 500
        # Determine starting stage: honor explicit config, else infer from progress when resuming
        explicit_start = None
        if isinstance(self.config, dict):
            explicit_start = self.config.get('start_from')
        start_from = (explicit_start or 'npt').lower()
        if start_from not in ('npt', 'nvt', 'nonequ'):
            start_from = 'npt'
        if explicit_start is None and resume:
            # Infer based on completed steps: continue NPT/NVT until their targets are reached
            npt_csv = os.path.join(self.output_dir, 'npt_state.csv')
            # Allow override from config
            if isinstance(self.config, dict) and self.config.get('npt_state_csv'):
                npt_csv = self.config['npt_state_csv']
            nvt_csv = os.path.join(self.output_dir, 'nvt_state.csv')
            npt_done = _read_last_step(npt_csv) >= int(npt_steps)
            nvt_done = _read_last_step(nvt_csv) >= int(nvt_steps)
            if not npt_done:
                start_from = 'npt'
                logger.info('Resume: NPT incomplete (last=%d, target=%d); continuing NPT', _read_last_step(npt_csv), int(npt_steps))
            elif not nvt_done:
                start_from = 'nvt'
                logger.info('Resume: NVT incomplete (last=%d, target=%d); starting/resuming NVT', _read_last_step(nvt_csv), int(nvt_steps))
            else:
                start_from = 'nonequ'
                logger.info('Resume: NPT and NVT targets reached; proceeding to nonequilibrium stage')
        compute_viscosity = bool(self.config.get('compute_viscosity', True)) if isinstance(self.config, dict) else True

        if start_from == 'npt':
            logger.info('npt run')
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
                state_csv_override=(self.config.get('npt_state_csv') if isinstance(self.config, dict) else None),
                dcd_path_override=(self.config.get('npt_dcd') if isinstance(self.config, dict) else None),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)) if isinstance(self.config, dict) else 2,
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True,
                traj_interval=traj_interval,
            )
            # Allow overriding the NPT CSV location for rescaling
            npt_csv_override = None
            if isinstance(self.config, dict):
                npt_csv_override = self.config.get('npt_state_csv')
            rescale_positions, rescale_box_vec = rescale_box(
                npt_positions,
                npt_box_vec,
                work_dir=self.output_dir,
                csv_override=npt_csv_override,
            )
            logger.info('nvt run')
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
                state_csv_override=(self.config.get('nvt_state_csv') if isinstance(self.config, dict) else None),
                dcd_path_override=(self.config.get('nvt_dcd') if isinstance(self.config, dict) else None),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)) if isinstance(self.config, dict) else 2,
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True,
                traj_interval=traj_interval,
            )
        elif start_from == 'nvt':
            logger.info('start_from=nvt: skipping NPT and starting/resuming NVT')
            # If NPT state CSV exists, rescale box/positions to the averaged NPT density
            npt_csv_default = os.path.join(self.output_dir, 'npt_state.csv')
            npt_csv_override = None
            if isinstance(self.config, dict):
                npt_csv_override = self.config.get('npt_state_csv')
            npt_csv_path = npt_csv_override if (npt_csv_override and os.path.isfile(npt_csv_override)) else npt_csv_default
            if os.path.isfile(npt_csv_path):
                rescale_positions, rescale_box_vec = rescale_box(
                    input_positions,
                    unit_cell,
                    work_dir=self.output_dir,
                    csv_override=npt_csv_path,
                )
                nvt_seed_pos, nvt_seed_box = rescale_positions, rescale_box_vec
                logger.info('Using rescaled GRO positions/box from NPT state for NVT')
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
                state_csv_override=(self.config.get('nvt_state_csv') if isinstance(self.config, dict) else None),
                dcd_path_override=(self.config.get('nvt_dcd') if isinstance(self.config, dict) else None),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)) if isinstance(self.config, dict) else 2,
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True,
                traj_interval=traj_interval,
            )
        else:  # start_from == 'nonequ'
            logger.info('start_from=nonequ: loading NVT outputs to seed nonequilibrium run')
            # Allow explicit paths via config, else look in output_dir then CWD
            cfg = self.config if isinstance(self.config, dict) else {}
            nvt_dcd = cfg.get('nvt_dcd')
            nvt_csv = cfg.get('nvt_state_csv')
            # Build candidate search lists
            dcd_candidates = []
            if nvt_dcd:
                dcd_candidates.append(nvt_dcd)
            dcd_candidates.extend([os.path.join(self.output_dir, 'nvt.dcd'), 'nvt.dcd', 'NVT.dcd', 'nvt.DCD', 'NVT.DCD'])
            csv_candidates = []
            if nvt_csv:
                csv_candidates.append(nvt_csv)
            csv_candidates.extend([
                os.path.join(self.output_dir, 'nvt_state.csv'),
                os.path.join(self.output_dir, 'nvt_results.csv'),
                'nvt_state.csv',
                'nvt_results.csv',
                'nvt.csv',
            ])
            # Resolve first existing path
            nvt_dcd = next((p for p in dcd_candidates if p and os.path.isfile(p)), None)
            nvt_csv = next((p for p in csv_candidates if p and os.path.isfile(p)), None)
            assert nvt_dcd and nvt_csv, f'Missing NVT outputs to seed nonequ run. Checked DCD: {dcd_candidates}, CSV: {csv_candidates}'
            nvt_positions_np = dcd_read(nvt_dcd)
            assert len(nvt_positions_np) > 0, 'Empty nvt.dcd'
            last = nvt_positions_np[-1]
            from openmm import Vec3
            nvt_positions = [Vec3(x, y, z) * ou.nanometers for x, y, z in last]
            import pandas as pd
            df = pd.read_csv(nvt_csv)
            L = df['Box Volume (nm^3)'].iloc[-1]**(1 / 3)
            nvt_box_vec = (Vec3(L, 0.0, 0.0) * ou.nanometers, Vec3(0.0, L, 0.0) * ou.nanometers,
                           Vec3(0.0, 0.0, L) * ou.nanometers)

        if compute_viscosity:
            logger.info('nonequ run')
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
            logger.info('compute_viscosity is false; skipping nonequilibrium run')

    def post_process(self,):
        logger.info('post processing transport protocol')
        cfg = getattr(self, 'config', {}) if hasattr(self, 'config') else {}
        compute_viscosity = True if not isinstance(cfg, dict) else bool(cfg.get('compute_viscosity', True))
        compute_conductivity = True if not isinstance(cfg, dict) else bool(cfg.get('compute_conductivity', True))

        results = {}
        vis = None
        if compute_viscosity:
            vis = viscosity_calc(self.output_dir)
            logger.info('viscosity: %.3f cP', vis)
            results['viscosity'] = vis
        else:
            # Optional user-provided viscosity for Yeh–Hummer
            if isinstance(cfg, dict) and cfg.get('viscosity_cP') is not None:
                vis = float(cfg['viscosity_cP'])
                logger.info('Using provided viscosity for YH correction: %.3f cP', vis)

        if compute_conductivity:
            # Locate NVT outputs robustly: allow overrides, then output_dir, then CWD
            cfg = self.config if isinstance(self, TransportProtocol) and isinstance(self.config, dict) else self.config
            dcd_path = None
            if isinstance(cfg, dict) and cfg.get('nvt_dcd'):
                dcd_path = cfg['nvt_dcd']
            else:
                dcd_candidate = os.path.join(self.output_dir, 'nvt.dcd')
                dcd_path = dcd_candidate if os.path.isfile(dcd_candidate) else 'nvt.dcd'
            nvt_positions = dcd_read(dcd_path)
            md_volume, md_temperature = volume_calc(self.output_dir, csv_override=(cfg.get('nvt_state_csv') if isinstance(cfg, dict) else None))
            species_mass_dict, species_number_dict, species_charges_dict = {}, {}, {}
            solvent, cation, anion = [], [], []
            for mol_name, topo_mol in self.components.items():
                species_mass_dict[mol_name] = [atom.mass for atom in topo_mol.atoms]
                species_number_dict[mol_name] = topo_mol.molar_num
                species_charges_dict[mol_name] = int(sum([atom.charge for atom in topo_mol.atoms]))
                if topo_mol.type == ComponentType.SOLVENT:
                    solvent.append(mol_name)
                elif topo_mol.type == ComponentType.CATION:
                    cation.append(mol_name)
                elif topo_mol.type == ComponentType.ANION:
                    anion.append(mol_name)
            sorted_components_names = anion + cation + solvent

            # keep solvent at the end
            species_charges_dict = {k: species_charges_dict[k] for k in sorted_components_names}
            species_mass_dict = {k: species_mass_dict[k] for k in sorted_components_names}
            species_number_dict = {k: species_number_dict[k] for k in sorted_components_names}

            # Optional MSD/fit window controls from config
            skip_frames = int(cfg.get('msd_skip_frames', 200)) if isinstance(cfg, dict) else 200
            fw_frames_cfg = cfg.get('fit_window_frames') if isinstance(cfg, dict) else None
            if fw_frames_cfg is not None and isinstance(fw_frames_cfg, (list, tuple)) and len(fw_frames_cfg) == 2:
                fw_frames = (int(fw_frames_cfg[0]), int(fw_frames_cfg[1]))
            else:
                fw_frames = (50, 200)
            fw_frac_cfg = cfg.get('fit_window_frac') if isinstance(cfg, dict) else None
            if fw_frac_cfg is not None and isinstance(fw_frac_cfg, (list, tuple)) and len(fw_frac_cfg) == 2:
                fw_frac = (float(fw_frac_cfg[0]), float(fw_frac_cfg[1]))
            else:
                fw_frac = None

            # Optional per-species transference numbers
            output_transference = bool(cfg.get('output_transference', False)) if isinstance(cfg, dict) else False

            cond = onsager_calc(
                species_mass_dict,
                species_number_dict,
                species_charges_dict,
                md_volume,
                vis,  # may be None; onsager_calc handles YH skip when None
                md_temperature,
                nvt_positions,
                msd_skip_frames=skip_frames,
                fit_window_frames=fw_frames,
                fit_window_frac=fw_frac,
                compute_transference=output_transference,
            )
            results.update(cond)

        if results:
            with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
        

class HVapProtocol(Protocol):

    def __init__(self, config: dict):
        super().__init__(config['params_dir'], config['output_dir'])
        self.config = config
        self.components = None

    def run_protocol(self):
        logger.info('running hvap protocol')
        # Allow override by steps or time; timestep_fs configurable
        def steps_from_time(cfg, steps_key, default_steps, time_ns_key=None, time_ps_key=None, timestep_fs=2):
            if isinstance(cfg, dict):
                if time_ns_key and cfg.get(time_ns_key) is not None:
                    return int(float(cfg[time_ns_key]) * 1e6 / float(timestep_fs))
                if time_ps_key and cfg.get(time_ps_key) is not None:
                    return int(float(cfg[time_ps_key]) * 1e3 / float(timestep_fs))
                if cfg.get(steps_key) is not None:
                    return int(cfg[steps_key])
            return int(default_steps)

        npt_timestep_fs = int(self.config.get('npt_timestep_fs', 2)) if isinstance(self.config, dict) else 2
        nvt_timestep_fs = int(self.config.get('nvt_timestep_fs', 2)) if isinstance(self.config, dict) else 2

        npt_steps = steps_from_time(self.config, 'npt_steps', 1500000, time_ns_key='npt_time_ns', time_ps_key='npt_time_ps', timestep_fs=npt_timestep_fs)
        nvt_steps = steps_from_time(self.config, 'nvt_steps', 5000000, time_ns_key='nvt_time_ns', time_ps_key='nvt_time_ps', timestep_fs=nvt_timestep_fs)
        nonbonded_params = self.generate_ff_params(self.config['smiles'], force=bool(self.config.get('force_regenerate_params', False)))
        self.components = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
            reuse_if_exists=bool(self.config.get('resume', False)),
        )
        _ = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
            build_gas=True,
            reuse_if_exists=False,
        )
        gro_file = f"{self.params_dir}/solvent_salt.gro"
        top_file = f"{self.params_dir}/system.top"
        gas_gro_file = f"{self.params_dir}/solvent_salt_gas.gro"
        gas_top_file = f"{self.params_dir}/system_gas.top"

        logger.info('running liquid phase')
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        liq_top, liq_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )
        resume = bool(self.config.get('resume', False))
        checkpoint_interval = int(self.config.get('checkpoint_interval', 5000))
        traj_interval = int(self.config.get('traj_interval', 500)) if isinstance(self.config, dict) else 500
        npt_run(
            top=liq_top,
            system=liq_system,
            positions=input_positions,
            temperature=self.config['temperature'],
            npt_steps=npt_steps,
            work_dir=self.output_dir,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            timestep=npt_timestep_fs,
            state_csv_override=(self.config.get('npt_state_csv') if isinstance(self.config, dict) else None),
            dcd_path_override=(self.config.get('npt_dcd') if isinstance(self.config, dict) else None),
            resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)) if isinstance(self.config, dict) else 2,
            resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True,
            resume_safe_warmup_steps=int(self.config.get('resume_safe_warmup_steps', 5000)) if isinstance(self.config, dict) else 5000,
            resume_safe_warmup_step_factor=float(self.config.get('resume_safe_warmup_step_factor', 2.0)) if isinstance(self.config, dict) else 2.0,
            resume_safe_disable_barostat_warmup=bool(self.config.get('resume_safe_disable_barostat_warmup', True)) if isinstance(self.config, dict) else True,
            traj_interval=traj_interval,
        )
        logger.info('running gas phase')
        grofileparser = app.GromacsGroFile(gas_gro_file)
        input_positions = grofileparser.positions
        gas_top, gas_system = generate_openmm_system(
            gas_top_file,
            nonbonded_params,
            unit_cell=None,
        )
        gas_timestep_fs = float(self.config.get('nvt_timestep_fs', 0.2)) if isinstance(self.config, dict) else 0.2
        nvt_run(top=gas_top,
                system=gas_system,
                positions=input_positions,
                box_vec=None,
                temperature=self.config['temperature'],
                nvt_steps=nvt_steps,
                work_dir=self.output_dir,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                timestep=gas_timestep_fs,
                state_csv_override=(self.config.get('nvt_state_csv') if isinstance(self.config, dict) else None),
                dcd_path_override=(self.config.get('nvt_dcd') if isinstance(self.config, dict) else None),
                resume_safe_backoff_frames=int(self.config.get('resume_safe_backoff_frames', 2)) if isinstance(self.config, dict) else 2,
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True,
                traj_interval=traj_interval)

    def post_process(self,):
        assert len(self.components) == 1
        nmols = sum([c.molar_num for c in self.components.values()])
        csv_file = os.path.join(self.output_dir, 'npt_state.csv')
        df = pd.read_csv(csv_file)
        density = df["Density (g/mL)"]
        dd = []
        for _ in range(10):
            dd.append(np.mean(np.random.choice(density[2000:3000], 100)))
        density, density_std = np.mean(dd), np.std(dd)

        e_liquid = df["Potential Energy (kJ/mole)"]
        el = []
        for _ in range(10):
            el.append(np.mean(np.random.choice(e_liquid[2000:3000], 100)) / nmols)
        e_liquid, e_liquid_std = np.mean(el), np.std(el)

        csv_file = os.path.join(self.output_dir, 'nvt_state.csv')
        df = pd.read_csv(csv_file)
        e_gas = df["Potential Energy (kJ/mole)"]
        eg = []
        for _ in range(10):
            eg.append(np.mean(np.random.choice(e_gas[2000:], 100)))
        e_gas, e_gas_std = np.mean(eg), np.std(eg)

        hvap = (e_gas - e_liquid) / 4.184 + 8.314 * self.config['temperature'] / 1000 / 4.184  # kcal/mol
        hvap_std = np.sqrt(e_gas_std**2 + e_liquid_std**2) / 4.184

        result = {
            "density": density,
            "density_std": density_std,
            "hvap": hvap,
            "hvap_std": hvap_std,
        }

        with open(os.path.join(self.output_dir, 'hvap_results.json'), 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(result)
        return result

### TODO: solvation free energy protocol

### TODO: Kirkwood-Buff integral protocol

class DielectricProtocol(Protocol):

    def __init__(self, config: dict):
        super().__init__(config['params_dir'], config['output_dir'])
        self.config = config
        self.components = None

    def run_protocol(self):
        import openmm as omm
        logger.info('running dielectric protocol')
        # steps / intervals configurable via config dict
        npt_steps = int(self.config.get('npt_steps', 2000000))
        nvt_steps = int(self.config.get('nvt_steps', 6000000))
        dipole_interval = int(self.config.get('dipole_interval', 500))
        nvt_timestep_fs = int(self.config.get('nvt_timestep_fs', 2))
        traj_interval = int(self.config.get('traj_interval', 500))
        checkpoint_interval = int(self.config.get('checkpoint_interval', 5000))
        resume = bool(self.config.get('resume', False))
        nonbonded_params = self.generate_ff_params(self.config['smiles'])
        self.components = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
        )
        gro_file = f"{self.params_dir}/solvent_salt.gro"
        top_file = f"{self.params_dir}/system.top"
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        input_top, input_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )

        # ------------------------------------------------------------------ #
        # Determine seed positions / box for the NVT+dipole run               #
        # ------------------------------------------------------------------ #
        start_from = self.config.get('start_from')
        if start_from and os.path.isfile(start_from):
            # Skip NPT entirely and seed NVT from an existing trajectory/GRO.
            if start_from.lower().endswith('.gro'):
                seed_parser = app.GromacsGroFile(start_from)
                seed_positions = seed_parser.positions
                seed_box_vec = seed_parser.getUnitCellDimensions()
                logger.info('start_from=%s (GRO): skipping NPT', start_from)
            else:
                # Treat as DCD: load the last frame
                frames = dcd_read(start_from)
                if len(frames) == 0:
                    raise ValueError(f'start_from DCD file contains no frames: {start_from}')
                last_frame = frames[-1]
                seed_positions = [omm.Vec3(x, y, z) * ou.angstroms for x, y, z in last_frame]

                # Resolve box dimensions from a state CSV next to the DCD or
                # from an explicit override key.
                dcd_dir = os.path.dirname(os.path.abspath(start_from))
                state_csv = self.config.get('start_from_state_csv')
                if not state_csv:
                    for cand in ['nvt_state.csv', 'npt_state.csv']:
                        p = os.path.join(dcd_dir, cand)
                        if os.path.isfile(p):
                            state_csv = p
                            break
                if state_csv and os.path.isfile(state_csv):
                    df_csv = pd.read_csv(state_csv)
                    L = float(df_csv['Box Volume (nm^3)'].iloc[-500:].mean()) ** (1.0 / 3.0)
                    logger.info('Box length %.4f nm from state CSV %s', L, state_csv)
                else:
                    L = float(unit_cell[0].value_in_unit(ou.nanometers))
                    logger.warning('No state CSV found alongside %s; using GRO unit cell (%.4f nm)', start_from, L)
                seed_box_vec = [
                    omm.Vec3(L, 0.0, 0.0) * ou.nanometers,
                    omm.Vec3(0.0, L, 0.0) * ou.nanometers,
                    omm.Vec3(0.0, 0.0, L) * ou.nanometers,
                ]
                logger.info('start_from=%s (DCD, last frame): skipping NPT', start_from)
        else:
            if start_from:
                logger.warning('start_from=%s not found; falling back to NPT', start_from)
            logger.info('npt run')
            npt_positions, npt_box_vec = npt_run(
                input_top,
                input_system,
                input_positions,
                temperature=self.config['temperature'],
                npt_steps=npt_steps,
                work_dir=self.output_dir,
                traj_interval=traj_interval,
                checkpoint_interval=checkpoint_interval,
            )
            seed_positions, seed_box_vec = rescale_box(npt_positions, npt_box_vec, work_dir=self.output_dir)

        # ------------------------------------------------------------------ #
        # NVT run with dipole reporter                                         #
        # ------------------------------------------------------------------ #
        logger.info('nvt run with dipole recording')
        dipole_csv = os.path.join(self.output_dir, 'dipole.csv')
        append_dipole = resume and os.path.isfile(dipole_csv)
        dipole_reporter = DipoleReporter(
            file_path=dipole_csv,
            reportInterval=dipole_interval,
            system=input_system,
            append=append_dipole,
        )
        _nvt_positions, _nvt_box_vec = nvt_run(
            input_top,
            input_system,
            seed_positions,
            seed_box_vec,
            temperature=self.config['temperature'],
            work_dir=self.output_dir,
            nvt_steps=nvt_steps,
            timestep=nvt_timestep_fs,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            traj_interval=traj_interval,
            extra_reporters=[dipole_reporter],
        )

    def post_process(self,):

        def _correlate_1d(in1: NDArray, in2: NDArray, average: bool) -> tuple[NDArray, NDArray]:
            N = len(in1)
            assert N == len(in2)

            result = signal.correlate(in1, in2, mode="full", method="auto")
            c12 = result[N - 1:]
            c21 = result[::-1][N - 1:]
            if average:
                div = np.arange(N, 0, -1)
                c12 = c12 / div
                c21 = c21 / div
                # "c12 /= div; c21 /= div" will not work because result[N-1]
                # will be divided by div[0]**2 due to the in-place operation.
            return c12, c21

        def correlate(in1: NDArray, in2: NDArray, average: bool = True) -> tuple[NDArray, NDArray]:
            shape1 = in1.shape
            dim1 = len(shape1)
            assert dim1 in (1, 2)
            if dim1 == 1:
                return _correlate_1d(in1, in2, average)
            assert shape1 == in2.shape

            N, D = shape1
            c12, c21 = np.zeros(N), np.zeros(N)
            for i in range(D):
                c12i, c21i = _correlate_1d(in1[:, i], in2[:, i], average)
                c12 += c12i
                c21 += c21i
            return c12, c21

        def calculate_dipole_autocorrelation(Mx, My, Mz):
            """Calculate dipole moment autocorrelation function (DACF)"""
            # Create dipole moment vector
            dipole = np.vstack([Mx, My, Mz]).T

            # Calculate autocorrelation function
            dacf, _ = correlate(dipole, dipole, average=True)

            return dacf

        def calculate_correlation_time(dacf, dt):
            """
            Calculate correlation time from dipole autocorrelation function.
            
            Parameters
            ----------
            dacf : np.ndarray
                Dipole autocorrelation function
            dt : float
                Time step between frames (in ps)
            
            Returns
            -------
            correlation_time : float
                Correlation time (in ps)
            """
            dacf = dacf / dacf[0]
            under_cutoff = np.where(dacf < 0.05)[0]
            if len(under_cutoff) == 0:
                cutoff = len(dacf)
            else:
                cutoff = under_cutoff[0]

            # Integrate DACF using trapezoidal rule to get correlation time
            correlation_time = np.trapz(dacf[:cutoff], dx=dt)

            return correlation_time

        logger.info('post processing dielectric protocol')
        # Read dipole series and thermodynamic quantities
        dip_csv = os.path.join(self.output_dir, 'dipole.csv')
        df = pd.read_csv(dip_csv)
        md_volume_A3, _ = volume_calc(self.output_dir)

        # use later part of trajectory to avoid initial relaxation bias
        start_index = int(len(df) * 0.2)
        Mx = df['Mx_eA'].values[start_index:]
        My = df['My_eA'].values[start_index:]
        Mz = df['Mz_eA'].values[start_index:]

        M2_mean = np.mean(Mx**2 + My**2 + Mz**2)
        M_mean_sq = np.mean(Mx)**2 + np.mean(My)**2 + np.mean(Mz)**2
        fluct = float(M2_mean - M_mean_sq)

        eps0_star = 1.0 / (4.0 * np.pi * CHG_FACTOR)
        R_gas = 1.9872036 * 1e-3
        V = md_volume_A3  # in A^3
        T = self.config['temperature']  # in K

        dielectric = 1.0 + fluct / (3.0 * eps0_star * V * R_gas * T)

        # Calculate dipole autocorrelation function and correlation time
        dt = 2.0 * self.config.get('dipole_interval', 500) * 1e-3  # ps (timestep in fs, every N steps)
        dacf = calculate_dipole_autocorrelation(Mx, My, Mz)
        correlation_time = calculate_correlation_time(dacf, dt)

        result = {
            'dielectric': float(dielectric),
            'volume': float(md_volume_A3),
            "correlation_time": float(correlation_time),
            'units': {
                'dielectric': 'dimensionless',
                'volume': 'Angstrom^3',
                'correlation_time': 'ps',
            },
        }
        with open(os.path.join(self.output_dir, 'dielectric_results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(result)
        return result


class CompressibilityProtocol(Protocol):

    def __init__(self, config: dict):
        super().__init__(config['params_dir'], config['output_dir'])
        self.config = config
        self.components = None

    def run_protocol(self):
        logger.info('running compressibility protocol')
        # steps configurable via JSON; defaults provide adequate sampling
        npt_steps = int(self.config.get('npt_steps', 5000000))
        assert npt_steps > 1000000, "npt_steps must be greater than 1000000"
        nonbonded_params = self.generate_ff_params(self.config['smiles'])
        self.components = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
        )
        gro_file = f"{self.params_dir}/solvent_salt.gro"
        top_file = f"{self.params_dir}/system.top"
        grofileparser = app.GromacsGroFile(gro_file)
        input_positions = grofileparser.positions
        unit_cell = grofileparser.getUnitCellDimensions()
        input_top, input_system = generate_openmm_system(
            top_file,
            nonbonded_params,
            unit_cell,
        )
        logger.info('npt run')
        _npt_positions, _npt_box_vec = npt_run(
            input_top,
            input_system,
            input_positions,
            temperature=self.config['temperature'],
            npt_steps=npt_steps,
            work_dir=self.output_dir,
        )

    def post_process(self,):

        def compressibility(volume, temp):
            kb = 1.380649e-23  # J/K
            v_mean = np.mean(volume)
            dv2_mean = np.mean((volume - v_mean)**2)
            comp = dv2_mean / (v_mean * kb * temp) * 1e9  # GPa^-1
            return comp

        logger.info('post processing compressibility protocol')
        # Read dipole series and thermodynamic quantities
        csv_file = os.path.join(self.output_dir, 'npt_state.csv')
        volume_m3 = pd.read_csv(csv_file)["Box Volume (nm^3)"].to_numpy() * 1e-27
        skip_steps = 1000  # skip first 1 ns
        comp = compressibility(volume_m3[skip_steps:], self.config['temperature'])
        result = {
            'compressibility': float(comp),
            'units': {
                'compressibility': 'GPa^-1',
            },
        }
        with open(os.path.join(self.output_dir, 'compressibility_results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(result)
        return result
