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

import logging
from operator import itemgetter
from typing import Dict, Iterable, List, Union

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists

logger = logging.getLogger(__name__)

##########################################
##           read-only information
##########################################


def show_debug_info(rkmol: Chem.Mol):
    rkmol.Debug()
    for idx, rda in enumerate(rkmol.GetAtoms()):
        print('atom', idx, rda.GetPropsAsDict())
    for idx, rdb in enumerate(rkmol.GetBonds()):
        print('bond', idx, rdb.GetPropsAsDict())
    return


def get_mol_formula(rkmol: Chem.Mol) -> str:
    return Chem.rdMolDescriptors.CalcMolFormula(rkmol)


def get_mol_mass(rkmol: Chem.Mol) -> float:
    '''return molecular mass in atomic unit (CH4=16.031300127999998)'''
    return Chem.rdMolDescriptors.CalcExactMolWt(rkmol, onlyHeavy=False)


def get_symm_sssr_atom_indices(rkmol: Chem.Mol, ring_size_limit: int = 0) -> List[Iterable]:
    '''
        1. SSSR is not well defined, according to this document
        https://www.rdkit.org/docs/GettingStartedInPython.html#the-sssr-problem
        GetSymmSSSR partially fixes that.
        2. return types of GetSymmSSSR and GetSSSR:
        https://github.com/rdkit/rdkit/issues/5395
    '''
    rings = Chem.rdmolops.GetSymmSSSR(rkmol)
    small_rings = []
    for ring in rings:
        _ring = set(ring)
        if ring_size_limit > 0 and len(_ring) <= ring_size_limit:
            small_rings.append(_ring)
    return small_rings


def check_in_same_sssr(atom_indices: List[Iterable], symm_sssr_atom_indices: List[Iterable]) -> List[bool]:
    '''WARNING: if not in the same sssr, they may still be in the same but larger ring'''
    atom_indices = [set(idxs) for idxs in atom_indices]
    result = [False for _ in atom_indices]
    symm_sssr_atom_indices = [set(idxs) for idxs in symm_sssr_atom_indices]
    for i, idxs in enumerate(atom_indices):
        for ring in symm_sssr_atom_indices:
            if idxs.issubset(ring):
                result[i] = True
                break

    return result


def check_small_ring_torsion(rkmol: Chem.Mol, atom_indices: List[Iterable]) -> List[bool]:
    '''mark False if a ring torsion has sp2-sp2 or sp2-sp3 hybridization'''
    result = [True for _ in atom_indices]
    for i, idxs in enumerate(atom_indices):
        bond = rkmol.GetBondBetweenAtoms(idxs[1], idxs[2])
        if bond.IsInRingSize(3) or bond.IsInRingSize(4) or bond.IsInRingSize(5):
            hyb_sum = 0
            for j in [1, 2]:
                hyb = rkmol.GetAtomWithIdx(idxs[j]).GetHybridization()
                if hyb == Chem.rdchem.HybridizationType.SP:
                    hyb_sum += 1
                elif hyb == Chem.rdchem.HybridizationType.SP2:
                    hyb_sum += 2
                elif hyb == Chem.rdchem.HybridizationType.SP3:
                    hyb_sum += 3
            if hyb_sum >= 5:
                result[i] = False

    return result


def get_tfd_propers(rkmol: Chem.Mol) -> List:
    assert isinstance(rkmol, Chem.Mol)
    rot_atom_pairs = rkmol.GetSubstructMatches(RotatableBondSmarts)
    torsions, _ = CalculateTorsionLists(rkmol)
    result = []
    for _ in torsions:
        tor = _[0][0]
        if (tor[1], tor[2]) in rot_atom_pairs or (tor[2], tor[1]) in rot_atom_pairs:
            result.append(tor)
    result.sort(key=itemgetter(0, 1, 2, 3))
    return result


def calc_rmsd(pred_mol: Chem.Mol, label_mol: Chem.Mol, return_map: bool = False):
    matches = label_mol.GetSubstructMatches(pred_mol, uniquify=0)
    if not matches:
        raise ValueError("pred mol does not match label mol")
    atom_maps = [list(enumerate(match)) for match in matches]

    lowest_rmsd = float('inf')
    lowest_map = None
    for atom_map in atom_maps:
        rmsd = AllChem.AlignMol(pred_mol, label_mol, atomMap=atom_map)
        lowest_rmsd = min(rmsd, lowest_rmsd)
        lowest_map = atom_map
    if return_map:
        return lowest_rmsd, lowest_map
    else:
        return lowest_rmsd


def calc_tfd(pred_mol: Chem.Mol, label_mol: Chem.Mol) -> Union[float, int]:
    '''Calculate torsion fingerprint deviation(TFD) developed by Rarey's group
       (J. Chem. Inf. Model., 52, 1499, 2012) between two molecules.
    '''
    try:
        tfd = TorsionFingerprints.GetTFDBetweenMolecules(pred_mol, label_mol)
    except IndexError:
        smi = Chem.MolToSmiles(label_mol)
        # this removes unnecessary H atoms
        smi = Chem.CanonSmiles(smi, useChiral=True)
        print("TFD cannot be calculated for mol: {}".format(smi))
        tfd = 0.
    return tfd


def get_aromatic_flags(mol: Chem.Mol) -> List[int]:
    aromatic_flags = []
    for atom in mol.GetAtoms():
        aromatic_flags.append(atom.GetIsAromatic())
    return aromatic_flags


def get_sum_absolute_formal_charges(mol: Chem.Mol) -> int:
    '''get total 'absolute' formal charges'''
    count = 0
    for at in mol.GetAtoms():
        count += abs(at.GetFormalCharge())
    return count


def get_nnz_formal_charges(mol: Chem.Mol) -> int:
    '''get number of atoms with non-zero formal charges'''
    count = 0
    for at in mol.GetAtoms():
        count += 1 if at.GetFormalCharge() != 0 else 0
    return count


def get_x1_nb(mol: Chem.Mol) -> Dict[int, int]:
    '''mol must have all Hs been added'''
    x1_nb = {}
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 1:
            continue
        aidx = atom.GetIdx()
        nb = atom.GetNeighbors()
        assert len(nb) == 1
        x1_nb[aidx] = nb[0].GetIdx()
    return x1_nb
