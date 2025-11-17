# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess

import ase
import h5py
from ase.units import Bohr

from bytemol.core import Molecule

QCHEM_TEMP = '''{mol_lines}
$rem
  JOBTYPE FORCE
  METHOD b3lyp
  BASIS def2-svpd
  DFT_D D3_BJ
  SCF_CONVERGENCE 11
  THRESH 14
  MAX_SCF_CYCLES 50
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  PURECART 1
$end

$archive
enable_archive = True !Turns on generation of Archive
$end
'''


def gen_molecule_lines(atoms, net_charge):
    assert atoms is not None
    assert net_charge is not None
    spin_multiplicity = 1
    lines = ["$molecule\n", f"  {net_charge} {spin_multiplicity}\n"]
    for symb, coord in zip(atoms.symbols, atoms.positions):
        line = "  {:3} {:12.8f}  {:12.8f}  {:12.8f}\n".format(symb, coord[0], coord[1], coord[2])
        lines.append(line)
    lines.extend(["$end\n"])
    return lines


def main(mapped_smiles, mol_name):
    mol = Molecule.from_mapped_smiles(
        mapped_smiles,
        nconfs=1,
        name=mol_name,
    )
    net_charge = int(sum(mol.formal_charges))
    atoms = mol.conformers[0].to_ase_atoms()
    mol_lines = gen_molecule_lines(atoms, net_charge)
    qc_input = QCHEM_TEMP.format(mol_lines="".join(mol_lines))
    qcin = f"./{mol_name}.qcin"
    with open(qcin, "w") as f:
        f.write(qc_input)
    command = (f"QCSCRATCH=./ geometric-optimize --nt 8 --converge set GAU "
               f"--engine qchem {qcin}")
    subprocess.run(command, shell=True, check=True)
    h5_file = f"./{mol_name}.tmp/run.d/qarchive.h5"
    with h5py.File(h5_file, 'r') as f:
        last_job_id = sorted(map(int, list(f['job'].keys())))
        last_job = f['job'][str(last_job_id[-1])]['sp']
        atoms = ase.Atoms(numbers=last_job['structure']['nuclei'][()],
                          positions=last_job['structure']['coordinates'][()] * Bohr)
        atoms.info['mapped_smiles'] = mapped_smiles
        atoms.info['name'] = mol_name
        ase.io.write(f"./{mol_name}.xyz", atoms)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mol_name",
        type=str,
        default="ACT",
    )
    parser.add_argument(
        "--mapped_smiles",
        type=str,
        default="[C:1]([C:2](=[O:3])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]",
    )
    # EC: [C:1]1([H:7])([H:8])[C:2]([H:9])([H:10])[O:3][C:4](=[O:5])[O:6]1
    args = parser.parse_args()
    main(args.mapped_smiles, args.mol_name)
