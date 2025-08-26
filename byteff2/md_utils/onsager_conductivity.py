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
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# correlation functions
def correlate_xy(in1: torch.Tensor, in2: torch.Tensor):
    # MyNoteEq: 1.2
    # MyNoteEq: B.5
    # MyNoteEq: B.7
    # MyNoteEq: B.11
    # in1: (nframes, _) or (nframes,); in2: (nframes, _) or (nframes,)
    N = len(in1)
    N1, D1 = 0, 0
    assert N == len(in2)
    dim1 = in1.shape
    if len(dim1) == 1:
        N1, = dim1
        D1 = 1
    elif len(dim1) == 2:
        N1, D1 = dim1
    dtype = in1.dtype
    assert N1 == N
    assert dim1 == in2.shape
    assert dtype == in2.dtype

    x = in1.to(torch.float64)
    y = in2.to(torch.float64)
    if D1 == 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    D = torch.einsum("ij,ij->i", x, y)

    cm1 = torch.cumsum(D, dim=0)
    cm2 = torch.cumsum(torch.flip(D, dims=(0,)), dim=0)
    Q = cm1[-1] * 2
    f1 = torch.zeros(N, dtype=dtype)
    f1[1:] = Q - cm1[:-1] - cm2[:-1]
    f1[0] = Q

    X = torch.fft.fft(x, n=2**(N * 2 - 1).bit_length(), dim=0)  # pylint: disable=not-callable
    Y = torch.fft.fft(y, n=2**(N * 2 - 1).bit_length(), dim=0)  # pylint: disable=not-callable

    X_Y_conj = torch.einsum("ij,ij->ij", X, torch.conj(Y))
    c12 = torch.fft.ifft(X_Y_conj, dim=0)  # pylint: disable=not-callable
    c12 = torch.sum((c12[:N]).real, dim=-1)

    X_conj_Y = torch.einsum("ij,ij->ij", torch.conj(X), Y)
    c21 = torch.fft.ifft(X_conj_Y, dim=0)  # pylint: disable=not-callable
    c21 = torch.sum((c21[:N]).real, dim=-1)

    div = torch.arange(N, 0, -1)
    ans = (f1 - c12 - c21) / div
    return ans.to(in1.dtype)


def polyfit(x: torch.Tensor, y: torch.Tensor, deg: int):
    X = torch.stack([x**i for i in range(deg + 1)], dim=1)
    return torch.linalg.lstsq(X, y.unsqueeze(1)).solution  # pylint: disable=not-callable


class OnsagerUnit:
    mol = 6.022_140_76e+23

    # time
    ps = 1.0
    fs = 1.e-3 * ps
    ns = 1.e+3 * ps
    s = 1.e+9 * ns

    # length
    nm = 1.
    angstrom = 1.e-1 * nm
    cm = 1.e+7 * nm
    m = 1.e+9 * nm

    # mass
    amu = 1.
    g = mol * amu
    kg = 1.e+3 * g

    # temperature
    K = 1.

    # electric current
    A = 1.

    # energy
    J = kg * (m / s)**2

    # electricity
    Coulomb = s * A
    e = 1.602_176_634e-19 * Coulomb
    Faraday = e * mol
    Siemens = Coulomb * A / J

    # heat capacity
    kB = 1.380_649e-23 * J / K  # Boltzmann constant J/K

    # pressure
    Pa = J / m**3

    # viscosity
    cP = 1.e-3 * Pa * s  # mPa*s

    # const
    pbc_xi = 2.837297  # unitless


def build_Delta_index(nsp: int):
    n = nsp - 1
    idxn, count = {}, 0
    for i in range(n):
        for j in range(n):
            idxn[(i, j)] = count
            count += 1
    return idxn, count


def build_B_index(nsp: int):
    n = nsp - 1
    idxn, count = {}, 0
    for i in range(n):
        for j in range(nsp):
            if i != j:
                idxn[(i, j)] = count
                count += 1
    return idxn, count


# L-related
def remove_center_of_mass_error(L: torch.Tensor, masses: torch.Tensor):
    # MyNoteEq: A.6
    # L and masses: n dim
    # returns: n dim
    assert L.dtype == masses.dtype
    totalM = torch.sum(masses)
    LM = torch.matmul(L, masses)
    MLM = torch.dot(masses, LM) / totalM**2
    LM /= totalM
    return L - LM.unsqueeze(1) - LM.unsqueeze(0) + MLM


def Lambda_to_ionic_conductivity(Lambda: torch.Tensor, charges: torch.Tensor, N: int, T: float, V: float):
    # V: angstrom**3
    # 10^-10 m^2/s -> mS/cm
    # MyNoteEq: 3.1
    # MyNoteEq: 1.7
    Lambda_si = Lambda * (1.e-10 * unit.m**2 / unit.s)
    L_si = Lambda_si * N / (V * unit.angstrom**3) / (unit.kB * T * unit.K)
    sigma_si = unit.e**2 * charges.matmul(L_si).matmul(charges) / (1.e-3 * unit.Siemens / unit.cm)
    return sigma_si


# conductivity
def Dself_to_ionic_conductivity(Dself: torch.Tensor, charges: torch.Tensor, counts: torch.Tensor, T: float, V: float):
    # V: angstrom**3
    # 10^-10 m^2/s -> mS/cm
    # MyNoteEq: 3.2
    # MyNoteEq: 1.7
    V_si = V * unit.angstrom**3
    kB_T = unit.kB * T * unit.K
    sigma_si = unit.e**2 * torch.sum(charges**2 * Dself * counts) * (1.e-10 * unit.m**2 / unit.s) / (kB_T * V_si)
    sigma_si /= (1.e-3 * unit.Siemens / unit.cm)
    return sigma_si


def Delta_matrices(L: torch.Tensor, xfrac: torch.Tensor):
    # MyNoteEq: 2.2
    # MyNoteEq: 2.6
    # L and xfrac: n dim
    # returns Delta and Delta**-1: n-1 dim
    nsp = len(xfrac)
    n = nsp - 1
    dtype = L.dtype
    assert dtype == xfrac.dtype
    dmat = torch.zeros((n, n), dtype=dtype)
    xn = xfrac[n]
    for i in range(n):
        xi = xfrac[i]
        for j in range(n):
            xj = xfrac[j]
            delta_ij = L[i, j] / xj - L[i, n] / xn
            psum = 0.0
            for k in range(0, nsp):
                psum += L[k, j] / xj - L[k, n] / xn
            delta_ij -= xi * psum
            dmat[i, j] = delta_ij

    pnorm = torch.inf
    cond = torch.linalg.cond(dmat, pnorm)  # pylint: disable=not-callable
    logger.info(f"Condition number of Delta {cond:.3f} (p-norm {pnorm})")
    inv = torch.linalg.inv(dmat)  # pylint: disable=not-callable
    return dmat, inv


def MS_matrix(B: torch.Tensor, xfrac: torch.Tensor):
    # MyNoteEq: 2.6
    # B: n-1 dim
    # xfrac: n-dim
    # returns MS matrix: n dim
    nsp = len(xfrac)
    n = nsp - 1
    dtype = B.dtype
    assert dtype == xfrac.dtype
    Dyet = torch.zeros((nsp, nsp), dtype=dtype)

    bidx, bcount = build_Delta_index(nsp)
    didx, dcount = build_B_index(nsp)
    assert bcount == dcount
    X = torch.zeros((bcount, bcount), dtype=dtype)

    for i in range(n):
        bii = bidx[(i, i)]
        xi = xfrac[i]
        din = didx[(i, n)]
        X[bii, din] += xi
        for j in range(n + 1):
            if j != i:
                xj = xfrac[j]
                dij = didx[(i, j)]
                X[bii, dij] += xj

    for i in range(n):
        xi = xfrac[i]
        for j in range(n):
            if j != i:
                bij = bidx[(i, j)]
                dij = didx[(i, j)]
                din = didx[(i, n)]
                X[(bij, dij)] -= xi
                X[(bij, din)] += xi

    pnorm = torch.inf
    cond = torch.linalg.cond(X, pnorm)  # pylint: disable=not-callable
    logger.info(f"Condition number of X {cond:.3f} (B=X.(1/Dyet), p-norm {pnorm})")
    Xinv = torch.linalg.inv(X)  # pylint: disable=not-callable

    dvec = torch.matmul(Xinv, B.flatten())
    for k, v in didx.items():
        i, j = k
        Dyet[i, j] = 1.0 / dvec[v].detach().item()
    Dyet[n, 0:n] = Dyet[0:n, n]
    return (Dyet + torch.transpose(Dyet, 0, 1)) * 0.5


def X_matrices(xfrac: torch.Tensor, masses: torch.Tensor):
    # MyNoteEq: 2.5
    # xfrac and masses: n dim
    # returns X and X**-1: both are n-1 dim
    nsp = len(xfrac)
    assert nsp == len(masses)
    dtype = xfrac.dtype
    assert dtype == masses.dtype

    n = nsp - 1
    idxn, dim = build_Delta_index(nsp)
    matrix = torch.zeros((dim, dim), dtype=dtype)

    xn = xfrac[n]
    mn = masses[n]
    for i in range(n):
        xi = xfrac[i]
        for j in range(n):
            xj = xfrac[j]
            ij = idxn[(i, j)]

            matrix[ij, ij] += 1. / xj  # ij
            for r in range(n):
                mr = masses[r]
                ir = idxn[(i, r)]
                matrix[ij, ir] += 1. / xn * mr / mn  # in
            for k in range(n):
                kj = idxn[(k, j)]
                matrix[ij, kj] -= xi / xj
                for r in range(n):
                    mr = masses[r]
                    kr = idxn[(k, r)]
                    matrix[ij, kr] -= xi / xn * mr / mn  # kn
            # nj and nn
            for r in range(n):
                mr = masses[r]
                rj = idxn[(r, j)]
                matrix[ij, rj] += xi / xj * mr / mn  # nj
                for s in range(n):
                    ms = masses[s]
                    rs = idxn[(r, s)]
                    matrix[ij, rs] += xi / xn * mr / mn * ms / mn  # nn

    pnorm = torch.inf
    cond = torch.linalg.cond(matrix, pnorm)  # pylint: disable=not-callable
    logger.info(f"Condition number of X {cond:.3f} (Delta=X.Lambda, p-norm {pnorm})")
    inv = torch.linalg.inv(matrix)  # pylint: disable=not-callable
    return matrix, inv


# YH correction
def fsc_self_diffusivity(T: float, viscosity: float, L: float) -> float:
    # MyNoteEq: 4.6
    # T [K]; viscosity [cP]; L [angstrom]
    # D [10^10 m^2/s]
    coeff = unit.kB * (T * unit.K) * unit.pbc_xi / (6 * torch.pi * (viscosity * unit.cP) * unit.angstrom)
    coeff /= (1.e-10 * unit.m**2 / unit.s)
    return coeff / L


def fsc_full_Lambda(dfsc: Union[float, torch.Tensor],
                    Xinv: torch.Tensor,
                    masses: torch.Tensor,
                    Gamma: torch.Tensor = None):
    # MyNoteEq: 4.3
    # MyNoteEq: 4.4
    # dfsc: float or n-1 dim
    # Xinv, Gamma: n-1 dim
    # masses: n dim
    # returns: n dim
    dtype = Xinv.dtype
    nsp = len(masses)
    n = nsp - 1
    n2, _ = Xinv.shape
    assert n2 == n * n

    if isinstance(dfsc, float):
        Dfsc = torch.eye(n, dtype=dtype) * dfsc
    else:
        Dfsc = dfsc
    if Gamma is None:
        gamma = torch.eye(n, dtype=dtype)
    else:
        gamma = Gamma
    assert dtype == Dfsc.dtype and dtype == Xinv.dtype and dtype == masses.dtype and dtype == gamma.dtype

    DG1_mat = torch.matmul(Dfsc, torch.linalg.inv(gamma))  # pylint: disable=not-callable
    DG1_vec = DG1_mat.reshape(n * n)
    x1DG1_vec = torch.matmul(Xinv, DG1_vec)
    Lfsc_small = x1DG1_vec.reshape(n, n)
    Lfsc_full = torch.zeros((nsp, nsp), dtype=dtype)

    Lfsc_full[0:n, 0:n] = Lfsc_small
    mn = masses[n]
    Lfsc_full[0:n, n] = -torch.matmul(Lfsc_small, masses[0:n]) / mn
    Lfsc_full[n, 0:n] = -torch.matmul(masses[0:n], Lfsc_small) / mn
    Lfsc_full[n, n] = masses[0:n].matmul(Lfsc_small).matmul(masses[0:n]) / mn**2
    return Lfsc_full


unit = OnsagerUnit()


def onsager_calc(species_mass, species_number, species_charge, volume_angstrom3, viscosity_cP, T_K, positions):

    dtype = torch.float64
    nsp = len(species_mass)
    mol_mass_list, atom_mfrac_list, gro_mass_list, natom_list = [], [], [], []
    gro_range_list = []
    for _, mass_list in species_mass.items():
        mol_mass = sum(mass_list)
        mol_mass_list.append(mol_mass)
        atom_mfrac_list.append([atomic_mass / mol_mass for atomic_mass in mass_list])
        natom_list.append(len(mass_list))
    gro_range0 = 0
    for reversed_key in reversed(species_mass.keys()):
        gro_mass_list.extend(species_mass[reversed_key] * species_number[reversed_key])
        gro_range_list.append([gro_range0, gro_range0 + len(species_mass[reversed_key]) * species_number[reversed_key]])
        gro_range0 += len(species_mass[reversed_key]) * species_number[reversed_key]
    gro_range_list.reverse()

    Masses = torch.tensor(mol_mass_list, dtype=dtype)
    AtomMFrac = [torch.tensor(l, dtype=dtype) for l in atom_mfrac_list]
    AtomMasses = torch.tensor(gro_mass_list, dtype=dtype)
    TotalAtomMass = AtomMasses.sum()
    NAtoms = torch.tensor(natom_list)
    SpeciesCounts = torch.tensor(list(species_number.values()))
    SpeciesCountsTotal = SpeciesCounts.sum().to(dtype)
    # XFrac = SpeciesCounts / SpeciesCountsTotal
    Charges = torch.tensor(list(species_charge.values()), dtype=dtype)
    AtomRanges = torch.tensor(gro_range_list)
    BoxVolume = volume_angstrom3
    BoxLen = BoxVolume**(1 / 3)
    logger.info(f"Inferred cubic box length [angstrom]: {BoxLen}")
    Viscosity = viscosity_cP
    RoomT = T_K

    np_xyz = positions[200:]
    nt_start, nt_end = 50, 200

    xu = torch.from_numpy(np_xyz[:, :, 0])
    yu = torch.from_numpy(np_xyz[:, :, 1])
    zu = torch.from_numpy(np_xyz[:, :, 2])
    origx = np.einsum("hi,i->h", xu.to(dtype), AtomMasses) / TotalAtomMass
    origy = np.einsum("hi,i->h", yu.to(dtype), AtomMasses) / TotalAtomMass
    origz = np.einsum("hi,i->h", zu.to(dtype), AtomMasses) / TotalAtomMass
    torch.set_printoptions(precision=8)

    dts_ps = torch.tensor([i for i in range(nt_start, nt_end)], dtype=dtype)

    times = torch.tensor([i for i in range(xu.shape[0])], dtype=dtype)
    logger.info(f"Use frames: {len(times)}")
    kmsd_i = torch.zeros(nsp, dtype=dtype)
    Rxt, Ryt, Rzt = [], [], []
    msd_self = {}
    for i, sp in enumerate(species_mass):
        x1 = xu[:, AtomRanges[i, 0]:AtomRanges[i, 1]].reshape(-1, SpeciesCounts[i], NAtoms[i]).to(dtype)
        y1 = yu[:, AtomRanges[i, 0]:AtomRanges[i, 1]].reshape(-1, SpeciesCounts[i], NAtoms[i]).to(dtype)
        z1 = zu[:, AtomRanges[i, 0]:AtomRanges[i, 1]].reshape(-1, SpeciesCounts[i], NAtoms[i]).to(dtype)

        # self-diffusivity
        cmsx1 = torch.einsum("hij,j->hi", x1, AtomMFrac[i]) - origx[:, None]
        cmsy1 = torch.einsum("hij,j->hi", y1, AtomMFrac[i]) - origy[:, None]
        cmsz1 = torch.einsum("hij,j->hi", z1, AtomMFrac[i]) - origz[:, None]

        msd1x, msd1y, msd1z = 0., 0., 0.
        for j in range(cmsx1.shape[1]):
            # print('NE', i, j)
            msd1x += correlate_xy(cmsx1[:, j], cmsx1[:, j])
            msd1y += correlate_xy(cmsy1[:, j], cmsy1[:, j])
            msd1z += correlate_xy(cmsz1[:, j], cmsz1[:, j])
        msd1r = msd1x + msd1y + msd1z
        msd_self_tensor = msd1r / SpeciesCounts[i]  # A^2
        msd_self[sp] = msd_self_tensor.tolist()

        _intercept, kslope = polyfit(dts_ps, msd1r[nt_start:nt_end], 1)
        kmsd_i[i] = kslope

        # onsager
        Rxt.append(torch.einsum("hij,j->h", x1, AtomMFrac[i]) - SpeciesCounts[i] * origx)
        Ryt.append(torch.einsum("hij,j->h", y1, AtomMFrac[i]) - SpeciesCounts[i] * origy)
        Rzt.append(torch.einsum("hij,j->h", z1, AtomMFrac[i]) - SpeciesCounts[i] * origz)

    kmsd_xy = torch.zeros((nsp, nsp), dtype=dtype)

    for i in range(nsp):

        for j in range(i, nsp):
            # print('Onsager', i, j)
            msd2x = correlate_xy(Rxt[i], Rxt[j])
            msd2y = correlate_xy(Ryt[i], Ryt[j])
            msd2z = correlate_xy(Rzt[i], Rzt[j])
            msd2r = msd2x + msd2y + msd2z

            _intercept, kslope = polyfit(dts_ps, msd2r[nt_start:nt_end], 1)
            kmsd_xy[i, j] = kslope

    for i in range(nsp):
        for j in range(i):
            kmsd_xy[i, j] = kmsd_xy[j, i]

    outdata = {}
    outdata["cubic_box_length_unit"] = "angstrom"
    outdata["cubic_box_length"] = BoxLen
    outdata["conductivity_unit"] = "mS/cm"

    # convert from angstrom**2/ps to 10^-10 m**2/s
    # self diffusivity: divided by 6*Ni
    # onsager: divided by 6*N
    Dself_MD = kmsd_i * (unit.angstrom**2 / unit.ps) / (1e-10 * unit.m**2 / unit.s) / (6 * SpeciesCounts)
    RawLambda = kmsd_xy * (unit.angstrom**2 / unit.ps) / (1e-10 * unit.m**2 / unit.s) / (6 * SpeciesCountsTotal)
    Lambda__md = remove_center_of_mass_error(RawLambda, Masses)
    # ionic conductivity
    sigma__o__md = Lambda_to_ionic_conductivity(Lambda__md, Charges, SpeciesCountsTotal, RoomT, BoxVolume)
    sigma_ne__md = Dself_to_ionic_conductivity(Dself_MD, Charges, SpeciesCounts, RoomT, BoxVolume)
    # Delta__md, B__md = Delta_matrices(Lambda__md, XFrac)  # Delta matrix
    # MS__md = MS_matrix(B__md, XFrac)  # MS matrix
    # Gamma_Dfick = torch.eye(nsp - 1, dtype=dtype)
    # DFick__md = torch.matmul(Delta__md, Gamma_Dfick)  # Dfick matrix

    outdata["conductivity_onsager"] = sigma__o__md.item()
    outdata["conductivity_NE"] = sigma_ne__md.item()
    outdata["diffusivity_unit"] = "10^-10 m^2/s"

    DYH1 = fsc_self_diffusivity(RoomT, Viscosity, BoxLen)
    DselfInf = Dself_MD + DYH1
    outdata["Dself_inf"] = DselfInf.tolist()

    logger.info(f"msd_self: {DselfInf.tolist()}")
    logger.info(f"conductivity: {sigma__o__md.item()}")

    return outdata
