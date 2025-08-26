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

import numpy as np

# from gromacs document
# The electric conversion factor f=1/(4 pi eps0)=
# 138.935458 kJ mol−1 nm e−2. = 332.0637141491396 kcal mol−1 A e−2.
# chg for 'charge'
CHG_FACTOR = 332.0637141491396

MAX_RING_SIZE = 8
MAX_CONNECTIVITY = 6
MAX_FORMAL_CHARGE = 2

SUPPORTED_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 3]  # H, C, N, O, F, P, S, Cl, Br, I, Li
Angstrom_per_Bohr = 1 / 1.88973  # Bohr to A

# use He hyper parameters for Li, since He and [Li+] are isoelectronic
V_FREE = np.array([7.9, 35.7, 27.0, 22.7, 18.6, 95.7, 77.0, 66.7, 97.0, 152.2, 89
                  ]) * Angstrom_per_Bohr**3  # Dictionary for free atomic volume (unit: A ** 3)
V_FREE = V_FREE.tolist()
ALPHA_FREE = np.array([4.5, 12.0, 7.4, 5.4, 3.8, 25.0, 19.6, 15.0, 20.0, 35.0, 1.38
                      ]) * Angstrom_per_Bohr**3  # Dictionary for free atomic polarizability (unit: A ** 3)
ALPHA_FREE = ALPHA_FREE.tolist()
C6_FREE = np.array([6.5, 46.6, 24.2, 15.6, 9.5, 185, 134, 94.6, 162, 385, 1.46
                   ]) * Angstrom_per_Bohr**6 * 627.509474  # Dictionary for free atomic C6 (unit: kcal/mol * A ** 6)
C6_FREE = C6_FREE.tolist()
RVDW_FREE = np.array([3.1, 3.59, 3.34, 3.19, 3.04, 4.01, 3.86, 3.71, 3.93, 4.17, 2.65
                     ]) * Angstrom_per_Bohr  # Dictionary for free atomic Rvdw (unit: A)
RVDW_FREE = RVDW_FREE.tolist()

ELEMENT_MAP = {at: i for i, at in enumerate(SUPPORTED_ELEMENTS)}
ATOMIC_ENERGY = [
    -314.090219448421, -23753.0364666938, -34258.534155417066, -47114.40179198107, -62594.32101200567,
    -214128.32342868324, -249798.4319745637, -288721.4770273567, -1614989.3719496096, -4342264.204824389, 0., 0., 0.
]  # kcal/mol


class BondOrder(Enum):
    single = 1.
    double = 2.
    triple = 3.
    aromatic = 1.5


class MMTerm(Enum):
    """Possible topology terms for classical force field"""
    bond = 2
    angle = 3
    proper = 4
    improper = 5


MMTERM_WIDTH = {
    MMTerm.bond: 2,
    MMTerm.angle: 3,
    MMTerm.proper: 4,
    MMTerm.improper: 4,
}

MM_TOPO_MAP = {
    MMTerm.bond: 'Bond',
    MMTerm.angle: 'Angle',
    MMTerm.proper: 'ProperTorsion',
    MMTerm.improper: 'ImproperTorsion',
}
