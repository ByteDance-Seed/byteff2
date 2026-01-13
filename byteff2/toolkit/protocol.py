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


def predict_density(component: dict):
    density = 0
    total_molar_ratio = 0
    solvent = [c for c in component.values() if c.type == ComponentType.SOLVENT]
    cation = [c for c in component.values() if c.type == ComponentType.CATION]
    anion = [c for c in component.values() if c.type == ComponentType.ANION]
    for c in solvent:
        density += c.density * c.molar_num
        total_molar_ratio += c.molar_num
    sol_density = density / total_molar_ratio
    sol_ratio = sol_density
    # Add cation and anion
    for c in cation:
        sol_density += min(c.density * c.molar_num / total_molar_ratio * sol_ratio, 0.5)
    for c in anion:
        sol_density += min(c.density * c.molar_num / total_molar_ratio * sol_ratio, 0.5)
    return round(sol_density, 2)


def search_mixture(mol_ratio, min_atoms, max_atoms, components):
    result = []
    num_atoms = np.array([len(component.atoms) for component in components.values()])
    atoms_ratio = mol_ratio * num_atoms
    uni_mol_ratio = mol_ratio / np.min(mol_ratio)
    uni_atom_count = int(sum(uni_mol_ratio * num_atoms))
    min_count = (min_atoms - 1) // uni_atom_count + 1
    max_count = (max_atoms - 1) // uni_atom_count + 1
    steps = max((max_count - min_count), 1)
    for i in range(min_count, max_count, steps):
        guess = np.round(uni_mol_ratio * i).astype(int)
        guess_count = int(sum(guess * num_atoms))
        mix = np.round(guess_count * atoms_ratio / np.sum(atoms_ratio) / num_atoms).astype(int)
        result.append(guess_count)
    total_atoms = result[0]
    mix = np.round(total_atoms * atoms_ratio / np.sum(atoms_ratio) / num_atoms).astype(int)
    return total_atoms, mix


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
    solvent = [c for c in components.values() if c.type == ComponentType.SOLVENT]
    cation = [c for c in components.values() if c.type == ComponentType.CATION]
    anion = [c for c in components.values() if c.type == ComponentType.ANION]
    script = GMXScript()
    script.add('cd "$(dirname "$0")" ')
    for i, c in enumerate(solvent):
        # Generate the box from the first component
        if i == 0:
            # Generate the box for solvent
            script.init_gro_box(f"{c.name}.gro", box)
            rest_molecules = c.molar_num - 1
            if rest_molecules:
                script.insert_molecules(f"{c.name}.gro", rest_molecules)
            continue
        script.insert_molecules(f"{c.name}.gro", c.molar_num)

    # Add cation and anion
    for c in cation:
        script.insert_molecules(f"{c.name}.gro", c.molar_num)
    for c in anion:
        script.insert_molecules(f"{c.name}.gro", c.molar_num)
    # Add run md run command
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
                resume_safe_minimize=bool(self.config.get('resume_safe_minimize', True)) if isinstance(self.config, dict) else True)

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
