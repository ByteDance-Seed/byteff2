import json
import os
import shutil
import subprocess
from enum import Enum

import ase.io as aio
import numpy as np
import openmm.app as app
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
    atoms_gro = mol.conformers[0].to_ase_atoms()
    atoms_gro.set_array('residuenames', np.array([mol.name] * mol.natoms))
    aio.write(save_path, atoms_gro)


class Protocol:

    def __init__(self, params_dir: str, output_dir: str):
        os.makedirs(params_dir, exist_ok=True)
        self.params_dir = params_dir
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def generate_ff_params(self, component_smiles: dict):
        model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
        model = load_model(os.path.dirname(model_dir))
        all_nb_params = {}

        for mol_name, smiles in component_smiles.items():
            logger.info(f'generating force field params for {mol_name}')
            mol = Molecule.from_smiles(smiles, nconfs=1)
            mol.name = mol_name
            metadata, params, tfs, mol = get_nb_params(model, mol)
            tfs.write_itp(f'{self.params_dir}/{mol.name}.itp', separated_atp=True)
            write_gro(mol, f'{self.params_dir}/{mol.name}.gro')
            with open(f'{self.params_dir}/{mol.name}.json', 'w') as f:
                json.dump(params, f, indent=2)
            with open(f'{self.params_dir}/{mol.name}_nb_params.json', 'w') as file:
                nb_params = {'metadata': metadata}
                json.dump(nb_params, file, indent=2)
            all_nb_params[mol_name] = params
            all_nb_params['metadata'] = metadata

        return all_nb_params

    def build_system(self, total_atoms: int, components_ratio: dict, working_dir: str, build_gas: bool = False):
        logger.info(f'building system for {components_ratio.keys()}')
        # read and parse topo files
        os.makedirs(working_dir, exist_ok=True)
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

        box = init_box
        for _ in range(5):
            generate_system_gro(components, working_dir, box)
            command = f'cd {working_dir} && bash -x run_gmx.sh'
            try:
                child = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                print("Script time out!")
                box *= 1.05
                continue
            if child.returncode != 0:
                print("Script gmx failed")
                raise subprocess.CalledProcessError(child.returncode, command)
            gro_file = os.path.join(working_dir, "solvent_salt.gro")
            with open(gro_file, "r") as f:
                gro_total_atoms = int(f.readlines()[1].strip().split()[0])
            if real_total_atoms > gro_total_atoms:
                box *= 1.05
            else:
                break
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
        nonbonded_params = self.generate_ff_params(self.config['smiles'])
        _ = self.build_system(
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

        npt_run(
            top=input_top,
            system=input_system,
            positions=input_positions,
            temperature=self.config['temperature'],
            npt_steps=1500000,
            work_dir=self.output_dir,
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
        npt_steps = 4000000
        nvt_steps = 10000000
        nonequ_steps = 1000000
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
        npt_positions, npt_box_vec = npt_run(
            input_top,
            input_system,
            input_positions,
            temperature=self.config['temperature'],
            npt_steps=npt_steps,
            work_dir=self.output_dir,
        )
        rescale_positions, rescale_box_vec = rescale_box(npt_positions, npt_box_vec, work_dir=self.output_dir)
        logger.info('nvt run')
        nvt_positions, nvt_box_vec = nvt_run(
            input_top,
            input_system,
            rescale_positions,
            rescale_box_vec,
            temperature=self.config['temperature'],
            work_dir=self.output_dir,
            nvt_steps=nvt_steps,
        )
        logger.info('nonequ run')
        nonequ_run(
            input_top,
            input_system,
            nvt_positions,
            nvt_box_vec,
            temperature=self.config['temperature'],
            work_dir=self.output_dir,
            nonequ_steps=nonequ_steps,
        )

    def post_process(self,):
        logger.info('post processing transport protocol')
        vis = viscosity_calc(self.output_dir)
        md_volume, md_temperature = volume_calc(self.output_dir)
        logger.info('viscosity: %.3f', vis)

        nvt_positions = dcd_read(os.path.join(self.output_dir, 'nvt.dcd'))
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

        results = onsager_calc(
            species_mass_dict,
            species_number_dict,
            species_charges_dict,
            md_volume,
            vis,
            md_temperature,
            nvt_positions,
        )
        results['viscosity'] = vis
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)


class HVapProtocol(Protocol):

    def __init__(self, config: dict):
        super().__init__(config['params_dir'], config['output_dir'])
        self.config = config
        self.components = None

    def run_protocol(self):
        logger.info('running hvap protocol')
        npt_steps = 1500000
        nvt_steps = 5000000
        nonbonded_params = self.generate_ff_params(self.config['smiles'])
        self.components = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
        )
        _ = self.build_system(
            self.config['natoms'],
            self.config['components'],
            self.config['working_dir'],
            build_gas=True,
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
        npt_run(
            top=liq_top,
            system=liq_system,
            positions=input_positions,
            temperature=self.config['temperature'],
            npt_steps=npt_steps,
            work_dir=self.output_dir,
        )
        logger.info('running gas phase')
        grofileparser = app.GromacsGroFile(gas_gro_file)
        input_positions = grofileparser.positions
        gas_top, gas_system = generate_openmm_system(
            gas_top_file,
            nonbonded_params,
            unit_cell=None,
        )
        nvt_run(top=gas_top,
                system=gas_system,
                positions=input_positions,
                box_vec=None,
                temperature=self.config['temperature'],
                nvt_steps=nvt_steps,
                work_dir=self.output_dir,
                timestep=0.2)

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
