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

import json
import os
import random

from rdkit import Chem

from bytemol.core import Molecule

random.seed(42)

functional_group_db = {
    # Acidic and Related
    'Carboxylic Acid': '[CX3](=O)[OX2H]',
    'Carboxylic Anhydride': '[#6]C(=O)OC(=O)[#6]',
    # Amine/Amide Derivatives
    'Amide': '[NX3][CX3](=[OX1])',
    'Primary Amine': '[NX3;H2;!$(NC=O);!$(N=O)]',
    'Secondary Amine': '[NX3;H1;!$(NC=O);!$(N=O)]',
    'Tertiary Amine': '[NX3;H0;!$(NC=O);!$(N=O)]',
    'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    # Carbonyls
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Ketone': '[#6][CX3](=O)[#6]',
    'Ester': '[#6][CX3](=O)[OX2][#6]',
    'Carbonate': '[#6][OX2][CX3](=[OX1])[OX2][#6]',
    # Alcohols and Ethers
    'Alcohol': '[#6X4]-[OX2H]',
    'Phenol': '[c]-[OX2H1]',
    'Ether': '[OD2]([#6X4])[#6X4]',
    # Sulfur Derivatives
    'Thiol': '[#16X2H]',
    'Sulfide': '[#16X2H0][!$(S=O)][!$(S=O)]',
    'Sulfoxide': '[#16X3](=[OX1])[#6]',
    'Thioether': '[#6][SX2][#6]',
    # Halogens
    'Alkyl Halide': '[#6X4]-[F,Cl,Br,I]',
    # Rings and Aromatics
    'Benzene': 'c1ccccc1',
    'Heterocycle': '[n,o]',
    # nitrogen
    'Azido': '[#7X2]~[#7X2]~[#7X1]',
    'Nitroso': '[#7X2](=O)[#6]',
    'Nitrile': '[#7X1]#[#6X2]',
    # unsaturated
    'Alkene': '[#6]=[#6]',
    'Alkyne': '[#6]#[#6]',
    # P, S
    'Sulfone': '[#8]=[#16X4]=[#8]',
    'Sulfoxide': '[#16X3](~[OX1])[#6]',
    'Phosphine oxide': '[#15X4]=[#8]'
}


def find_functional_groups_from_db(smiles_string):
    """
    Identifies functional groups in a molecule using a more extensive
    internal library of SMARTS patterns.
    """

    mol = Chem.MolFromSmiles(smiles_string)
    if not mol:
        return ["Invalid SMILES"]

    found_groups = []
    for group_name, smarts in functional_group_db.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            found_groups.append(group_name)

    charge = 0
    symbols = set()
    for atom in mol.GetAtoms():
        charge += atom.GetFormalCharge()
        symbols.add(atom.GetSymbol())

    if charge != 0:
        found_groups.append('Ion')

    if symbols.issubset({'C', 'H'}) and not found_groups:
        found_groups.append('Alkanes')

    return found_groups


fp = 'trained_mols.json'

with open(fp) as file:
    name_smiles = json.load(file)

count = 0
records = {k: [] for k in functional_group_db.keys()}
records['Ion'] = []
records['Alkanes'] = []
for name, smiles in name_smiles.items():

    # skip wrong identification by rdkit
    if name == 'VC':
        continue

    identified_groups = find_functional_groups_from_db(smiles)
    for group in identified_groups:
        records[group].append((name, smiles))

save_dir = 'functional_group_examples'
os.makedirs(save_dir, exist_ok=True)

for k, v in records.items():
    if len(v) > 3:
        v = random.sample(v, 3)
    ss = []
    for i, (name, mps) in enumerate(v):
        mol = Molecule.from_mapped_smiles(mps)
        ss.append(mol.get_smiles())
        mol.to_image(f'{save_dir}/{k}_{i}.png', remove_h=True)
    print(k, ss)
