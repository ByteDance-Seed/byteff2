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

import glob
import json
import re
import subprocess

from bytemol.core import Molecule

QCHEM_TEMP = '''{mol_lines}
$rem
  JOBTYPE eda
  EDA2 1
  METHOD WB97M-V
  BASIS def2-tzvpd
  SCF_CONVERGENCE 11
  THRESH 14
  MAX_SCF_CYCLES 50
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  FD_MAT_VEC_PROD false
$end
$archive
enable_archive = True !Turns on generation of Archive
$end
'''


def gen_dimer_lines(atoms, net_charge):
    assert atoms is not None
    assert net_charge is not None
    spin_multiplicity = 1
    lines = ["$molecule\n", f"{sum(net_charge)} {spin_multiplicity}\n"]
    for atoms, net_charge in zip(atoms, net_charge):
        lines.append('--\n')
        lines.append(f'{net_charge} 1\n')
        for symb, coord in zip(atoms.symbols, atoms.positions):
            line = "  {:3} {:12.8f}  {:12.8f}  {:12.8f}\n".format(symb, coord[0], coord[1], coord[2])
            lines.append(line)
    lines.extend(["$end\n"])
    return lines


def parse_log(log_path):
    marker = '        Results of EDA2         '
    begin = None
    with open(log_path) as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line[:len(marker)] == marker:
                begin = i
                break

    text = ''.join(lines[begin:begin + 50])

    # Dictionary to store the extracted values
    values = {}

    # Regular expression patterns to capture values with their names
    pattern1 = re.compile(r'(\b[A-Z\s]+)\s*(-?\d+\.\d+)\b')
    pattern2 = re.compile(r'(\b[A-Z\s]+)=\s*(-?\d+\.\d+)')

    # Regular expression patterns to extract ELEC and CLS ELEC values
    elec_pattern = r"E_elec\s+\(ELEC\)\s+\(kJ/mol\)\s+=\s+([-+]?\d*\.\d+)"
    cls_elec_pattern = r"E_cls_elec\s+\(CLS ELEC\)\s+\(kJ/mol\)\s+=\s+([-+]?\d*\.\d+)"

    # Search for ELEC and CLS ELEC values
    elec_match = re.search(elec_pattern, text)
    cls_elec_match = re.search(cls_elec_pattern, text)

    # Extract and print the values
    if elec_match:
        elec_value = float(elec_match.group(1))
        values['ELEC'] = elec_value

    if cls_elec_match:
        cls_elec_value = float(cls_elec_match.group(1))
        values['CLS ELEC'] = cls_elec_value

    # Find all matches for pattern1
    matches1 = pattern1.findall(text)
    for match in matches1:
        name, value = match[0].strip(), match[1]
        if name:
            values[name] = float(value)

    # Find all matches for pattern2
    matches2 = pattern2.findall(text)
    for match in matches2:
        name, value = match[0].strip(), match[1]
        if name:
            values[name] = float(value)

    values['ELEC_PAULI'] = values.pop('PAULI')

    assert abs(values['ELEC_PAULI'] + values['DISP'] - values['FROZEN']) < 1e-3
    assert abs(values['PREPARATION'] + values['FROZEN'] + values['POLARIZATION'] + values['CHARGE TRANSFER'] -
               values['TOTAL']) < 1e-3

    # convert ot kcal/mol
    for k in values.keys():
        values[k] = values[k] / 4.184
    return values


def main(input_dir):
    mols = [Molecule.from_xyz(xyz) for xyz in glob.glob(f'{input_dir}/*.xyz')]
    assert len(mols) == 2
    atoms, net_charges, names = [], [], []
    for mol in mols:
        atoms.append(mol.conformers[0].to_ase_atoms())
        net_charges.append(int(sum(mol.formal_charges)))
        names.append(mol.name)
    dimer_lines = gen_dimer_lines(atoms, net_charges)
    qc_input = QCHEM_TEMP.format(mol_lines="".join(dimer_lines))
    qcin = f"./{names[0]}_{names[1]}.qcin"
    with open(qcin, 'w') as f:
        f.write(qc_input)
    command = f"QCSCRATCH=./ qchem -nt 32 -archive {qcin} dimer.out"
    subprocess.run(command, shell=True, check=True)
    eda_results = parse_log('dimer.out')
    with open(f'{names[0]}_{names[1]}.json', 'w') as f:
        json.dump(eda_results, f, indent=4)  # return results in kcal/mol


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='ACT_EC_dimer/conf_0')
    args = parser.parse_args()
    main(args.input_dir)
