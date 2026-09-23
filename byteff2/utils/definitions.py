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

from enum import Enum

# from gromacs document
# The electric conversion factor f=1/(4 pi eps0)=
# 138.935458 kJ mol−1 nm e−2. = 332.0637141491396 kcal mol−1 A e−2.
# chg for 'charge'
CHG_FACTOR = 332.0637141491396
fudgeLJ = 0.5
fudgeQQ = 0.8333333333

MAX_RING_SIZE = 8
MAX_CONNECTIVITY = 6
MAX_FORMAL_CHARGE = 2
PROPERTORSION_TERMS = 4

ATOMIC_ELEMENT_NUMBER_MAP = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
}
ATOMIC_NUMBER_ELEMENT_MAP = {v: k for k, v in ATOMIC_ELEMENT_NUMBER_MAP.items()}
SUPPORTED_ELEMENTS_NUM = len(ATOMIC_ELEMENT_NUMBER_MAP)

Angstrom_per_Bohr = 1 / 1.88973  # Bohr to A
# use He hyper parameters for Li, since He and [Li+] are isoelectronic
SUPPORT_V_FREE = {
    "H": 7.9,
    "C": 35.7,
    "N": 27.0,
    "O": 22.7,
    "F": 18.6,
    "P": 95.7,
    "S": 77.0,
    "Cl": 66.7,
    "Br": 97.0,
    "I": 152.2,
    "Li": 89.0,
    "B": 47.1,
    "Si": 99.5,
    "Na": 114.3,
    "K": 219.4,
}
V_FREE = list(
    {v: SUPPORT_V_FREE.get(k, 0.0) * Angstrom_per_Bohr**3 for k, v in ATOMIC_ELEMENT_NUMBER_MAP.items()}.values()
)

SUPPORT_ALPHA_FREE = {
    "H": 4.5,
    "C": 12.0,
    "N": 7.4,
    "O": 5.4,
    "F": 3.8,
    "P": 25.0,
    "S": 19.6,
    "Cl": 15.0,
    "Br": 20.0,
    "I": 35.0,
    "Li": 1.38,
    "B": 21.0,
    "Si": 37.0,
    "Na": 2.67,
    "K": 11.1,
}  # Dictionary for free atomic polarizability (unit: A ** 3)
ALPHA_FREE = list(
    {v: SUPPORT_ALPHA_FREE.get(k, 0.0) * Angstrom_per_Bohr**3 for k, v in ATOMIC_ELEMENT_NUMBER_MAP.items()}.values()
)

SUPPORT_C6_FREE = {
    "H": 6.5,
    "C": 46.6,
    "N": 24.2,
    "O": 15.6,
    "F": 9.5,
    "P": 185,
    "S": 134,
    "Cl": 94.6,
    "Br": 162,
    "I": 385,
    "Li": 1.46,
    "B": 99.5,
    "Si": 305.0,
    "Na": 6.38,
    "K": 64.3,
}  # Dictionary for free atomic C6 (unit: kcal/mol * A ** 6)
C6_FREE = list(
    {
        v: SUPPORT_C6_FREE.get(k, 0.0) * Angstrom_per_Bohr**6 * 627.509474 for k, v in ATOMIC_ELEMENT_NUMBER_MAP.items()
    }.values()
)

SUPPORT_RVDW_FREE = {
    "H": 3.1,
    "C": 3.59,
    "N": 3.34,
    "O": 3.19,
    "F": 3.04,
    "P": 4.01,
    "S": 3.86,
    "Cl": 3.71,
    "Br": 3.93,
    "I": 4.17,
    "Li": 2.65,
    "B": 3.89,
    "Si": 4.2,
    "Na": 2.91,
    "K": 3.55,
}  # Dictionary for free atomic Rvdw (unit: A)
RVDW_FREE = list(
    {v: SUPPORT_RVDW_FREE.get(k, 0.0) * Angstrom_per_Bohr for k, v in ATOMIC_ELEMENT_NUMBER_MAP.items()}.values()
)


class BondOrder(Enum):
    single = 1.0
    double = 2.0
    triple = 3.0
    aromatic = 1.5


class MMTerm(Enum):
    """Possible topology terms for classical force field"""

    bond = 2
    angle = 3
    proper = 4
    improper = 5


class NBMMTerm(Enum):
    nonbonded14 = 14
    nonbonded_all = 15


class MMParam(Enum):
    bond_k = 0
    bond_r0 = 1
    angle_k = 2
    angle_d0 = 3
    proper_k = 4
    improper_k = 6


class NBMMParam(Enum):
    sigma = 0
    epsilon = 1
    charges = 2


MMTERM_WIDTH = {
    MMTerm.bond: 2,
    MMTerm.angle: 3,
    MMTerm.proper: 4,
    MMTerm.improper: 4,
}

MM_TOPO_MAP = {
    MMTerm.bond: "Bond",
    MMTerm.angle: "Angle",
    MMTerm.proper: "ProperTorsion",
    MMTerm.improper: "ImproperTorsion",
}
