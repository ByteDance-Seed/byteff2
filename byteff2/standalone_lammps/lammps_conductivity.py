#!/usr/bin/env python3
"""
Compute ionic conductivity from LAMMPS trajectories using MDAnalysis.

Methods
- Nernst–Einstein (NE): from self-diffusion coefficients of charged species
  estimated from molecule center-of-mass mean-squared displacement (MSD).
- Onsager: builds cross-correlation matrix of species COM displacements to obtain
  the Onsager transport coefficients and conductivity.

Assumptions and requirements
- LAMMPS data/topology contains per-atom charges and masses.
- Dump/trajectory contains atom positions for multiple frames; velocities not required.
- Molecule IDs are present (molecule-ID in LAMMPS), allowing grouping atoms into ions.
- Coordinates are unwrapped (or we unwrap with MDAnalysis transformation).

Units and conversions
- Positions are in Angstrom from MDAnalysis; time step supplied in ps via --dt-ps.
- Volume is taken per-frame from trajectory box dimensions (Angstrom^3) and averaged.
- Temperature in K is required.
- Conductivity is reported in mS/cm.

Example
  python lammps_conductivity.py \
    --data system.data --dump traj.dump \
    --dt-ps 0.01 --temperature 298 \
    --unwrap --fit-window 0.6 0.95 --mode onsager --viscosity-cp 1.0

Notes
- This implementation uses a single-origin MSD per molecule. For best accuracy, use long
  trajectories. An FFT-based time-averaged kMSD is used for both NE and Onsager
  to mirror md_utils/onsager_conductivity.py behavior and improve statistics.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _require_mdanalysis():
    try:
        import MDAnalysis as mda  # noqa: F401
        from MDAnalysis.transformations.wrap import unwrap  # noqa: F401
        return True
    except Exception as exc:  # pylint: disable=broad-except
        sys.stderr.write(
            "MDAnalysis is required. Install via: pip install MDAnalysis\n"
            f"Import error: {exc}\n"
        )
        return False


def _volume_from_dims(dimensions: np.ndarray) -> float:
    """Compute volume from MDAnalysis ``ts.dimensions``.

    dimensions = [a, b, c, alpha, beta, gamma] with angles in degrees.
    Returns volume in Angstrom^3.
    """
    a, b, c, alpha_deg, beta_deg, gamma_deg = dimensions[:6]
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)
    vol = a * b * c * np.sqrt(
        1.0
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )
    # For orthorhombic cells, the sqrt() term is 1, reducing to a*b*c.
    return float(vol)


@dataclass
class Species:
    charge_int: int  # integer net charge per molecule for this species (0 for neutrals)
    mol_ids: List[int]


def identify_species(
    mol_to_charge: Dict[int, float],
    charge_tol: float = 0.25,
) -> Tuple[Dict[Tuple[int, int], Species], Species | None]:
    """Group molecules into species by integer net charge per molecule.

    - Charged species: dict keyed by (qi, sign) with qi in Z minus {0}.
    - Neutral group: a single Species with charge_int == 0 containing all near-zero molecules,
      or None if no neutrals.
    """
    grouped: Dict[int, List[int]] = defaultdict(list)
    neutrals: List[int] = []
    for mol_id, q in mol_to_charge.items():
        if abs(q) < charge_tol:
            neutrals.append(mol_id)
            continue
        qi = int(np.sign(q) * round(abs(q)))
        if qi == 0:
            neutrals.append(mol_id)
            continue
        grouped[qi].append(mol_id)

    species: Dict[Tuple[int, int], Species] = {}
    for qi, ids in grouped.items():
        species[(qi, int(np.sign(qi)))] = Species(charge_int=qi, mol_ids=sorted(ids))
    neutral_species = Species(charge_int=0, mol_ids=sorted(neutrals)) if len(neutrals) > 0 else None
    return species, neutral_species


def compute_com(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute center of mass for a set of atoms.

    positions: (n_atoms, 3) in Angstrom
    masses: (n_atoms,) in amu
    Returns: (3,) in Angstrom
    """
    msum = np.sum(masses)
    if msum <= 0:
        return np.average(positions, axis=0)
    return np.einsum("i,ij->j", masses, positions) / msum


def linear_fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope of linear fit y = a + b x using least squares."""
    A = np.vstack([np.ones_like(x), x]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[1])


def correlate_xy_np(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """Unbiased correlation used for kMSD, mirroring md_utils/onsager_conductivity.py.

    Returns ans[τ] ≈ E_t[(x(t+τ)-x(t))(y(t+τ)-y(t))] for τ=0..N-1.
    """
    x = np.asarray(in1, dtype=np.float64).reshape(-1)
    y = np.asarray(in2, dtype=np.float64).reshape(-1)
    assert x.shape == y.shape
    n = x.shape[0]
    # f1 term from cumulative sums of elementwise products
    D = x * y
    cm1 = np.cumsum(D)
    cm2 = np.cumsum(D[::-1])
    Q = cm1[-1] * 2.0
    f1 = np.empty(n, dtype=np.float64)
    f1[0] = Q
    if n > 1:
        f1[1:] = Q - cm1[:-1] - cm2[:-1]
    # FFT-based cross-correlations
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.fft(x, n=nfft)
    Y = np.fft.fft(y, n=nfft)
    c12 = np.fft.ifft(X * np.conj(Y)).real[:n]
    c21 = np.fft.ifft(np.conj(X) * Y).real[:n]
    # unbiased divide by number of pairs per lag
    div = np.arange(n, 0, -1, dtype=np.float64)
    ans = (f1 - c12 - c21) / div
    return ans.astype(np.float64)


def remove_com_error(L: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Remove center-of-mass contribution from Onsager matrix.

    Implements L - u v^T - v u^T + s, where u = L m / M, s = (m^T L m) / M^2.
    """
    masses = np.asarray(masses, dtype=float)
    L = np.asarray(L, dtype=float)
    M = float(np.sum(masses))
    u = L @ masses
    u /= M
    s = float(masses @ (L @ masses)) / (M * M)
    return L - u[:, None] - u[None, :] + s


def nernst_einstein_conductivity(
    D_species: Dict[int, float],
    N_species: Dict[int, int],
    T_K: float,
    volume_angstrom3: float,
) -> float:
    """Compute NE ionic conductivity in mS/cm.

    D_species: map charge_int -> self-diffusion in 1e-10 m^2/s (per species)
    N_species: map charge_int -> number of molecules of that species
    T_K: temperature [K]
    volume_angstrom3: average simulation volume [Angstrom^3]
    """
    # Physical constants
    kB = 1.380_649e-23  # J/K
    e = 1.602_176_634e-19  # C

    V_m3 = volume_angstrom3 * 1e-30  # A^3 -> m^3
    D_m2_s_1e10 = np.array([D_species[q] for q in D_species])
    Qs = np.array([q for q in D_species])
    Ns = np.array([N_species[q] for q in D_species], dtype=float)

    # Convert 1e-10 m^2/s units to m^2/s
    D_m2_s = D_m2_s_1e10 * 1e-10
    sigma_S_m = (e**2 / (kB * T_K * V_m3)) * np.sum((Qs**2) * Ns * D_m2_s)
    #sigma_mS_cm = sigma_S_m / 10.0  # 1 S/m = 10 mS/cm
    sigma_mS_cm = sigma_S_m * 10.0  # 1 S/m = 10 mS/cm
    return float(sigma_mS_cm)


def yeh_hummer_delta_D_1e10(T_K: float, viscosity_cP: float, box_len_A: float) -> float:
    """Yeh–Hummer finite-size correction for self-diffusivity in 1e-10 m^2/s units.

    ΔD = kB T ξ / (6 π η L), with ξ≈2.837297. Input η in cP, L in Angstrom.
    Output in 1e-10 m^2/s.
    """
    if viscosity_cP is None or viscosity_cP <= 0 or not np.isfinite(viscosity_cP):
        return 0.0
    kB = 1.380_649e-23
    xi = 2.837297
    eta_Pa_s = viscosity_cP / 1000.0
    L_m = box_len_A * 1e-10
    dD_SI = (kB * T_K * xi) / (6.0 * np.pi * eta_Pa_s * L_m)  # m^2/s
    return float(dD_SI / 1e-10)  # to 1e-10 m^2/s


def main():
    if not _require_mdanalysis():
        sys.exit(2)

    import MDAnalysis as mda
    from MDAnalysis.transformations.wrap import unwrap

    ap = argparse.ArgumentParser(description="Ionic conductivity (NE) from LAMMPS trajectory")
    ap.add_argument("--data", default=None, help="Optional LAMMPS data/topology file; if omitted, use dump-only")
    ap.add_argument("--dump", required=True, help="LAMMPS dump/trajectory file with positions across frames")
    ap.add_argument("--dump-format", default=None, help="Optional explicit MDAnalysis format, e.g., LAMMPSDUMP")
    ap.add_argument("--dt-ps", type=float, required=True, help="Time between frames in picoseconds")
    ap.add_argument("--temperature", type=float, required=True, help="Temperature in Kelvin")
    ap.add_argument("--unwrap", action="store_true", help="Unwrap PBC before analysis")
    ap.add_argument("--fit-window", nargs=2, type=float, default=(0.5, 0.9),
                   metavar=("START_FRAC", "END_FRAC"),
                   help="Fractional window over lags for slope fit (default: 0.5 0.9)")
    ap.add_argument("--skip-frames", type=int, default=200,
                   help="Discard this many initial frames before analysis (default: 200, like md_utils)")
    ap.add_argument("--fit-window-frames", nargs=2, type=int, default=None,
                   metavar=("START", "END"),
                   help="Explicit lag window [start,end) in frames for slope fit; overrides --fit-window. If omitted, uses --fit-window unless left at default, in which case [50,200] frames is used.")
    ap.add_argument("--group-neutrals-by-resname", action="store_true",
                   help="If topology provides residue names, split neutrals by resname instead of pooling them.")
    ap.add_argument("--mode", choices=["ne", "onsager"], default="ne",
                   help="Method: 'ne' (default) or 'onsager'")
    ap.add_argument("--viscosity-cp", type=float, default=None,
                   help="Optional viscosity (cP) to apply Yeh–Hummer correction to NE D_self")
    ap.add_argument("--print-lambda", action="store_true",
                   help="If set, include Lambda (10^-10 m^2/s) in JSON output (Onsager mode only)")
    ap.add_argument("--print-L-hat", action="store_true",
                   help="If set, include L_hat (1/(J s m)) in JSON output (Onsager mode only)")
    ap.add_argument("--compute-transference", action="store_true",
                   help="If set, compute and include species transference numbers t_i using Onsager matrix (Onsager mode only)")
    ap.add_argument("--json-out", default=None, help="Optional path to write results JSON")

    args = ap.parse_args()
    if not (0.0 <= args.fit_window[0] < args.fit_window[1] <= 1.0):
        ap.error("--fit-window must satisfy 0 <= start < end <= 1")

    # Load trajectory
    if args.data is not None:
        u = mda.Universe(args.data, args.dump, format=args.dump_format)
    else:
        # dump-only mode
        u = mda.Universe(args.dump, format=args.dump_format)
    if args.unwrap:
        u.trajectory.add_transformations(unwrap(u.atoms))

    # Build molecule index lists from one of: atoms.molnums, dump 'mol' column, or residues
    mol_to_indices: Dict[int, np.ndarray] = {}
    have_molnums = hasattr(u.atoms, "molnums")
    used_source = None
    if have_molnums:
        try:
            for mol_id, group in u.atoms.groupby("molnums").items():
                mol_to_indices[int(mol_id)] = group.indices
            used_source = "molnums"
        except Exception:
            mol_to_indices = {}
    if not mol_to_indices:
        # Try first frame 'mol' column from dump
        try:
            ts0 = u.trajectory[0]
            if hasattr(ts0, "data") and ts0.data is not None and "mol" in ts0.data:
                molarr = np.asarray(ts0.data["mol"], dtype=int)
                for mid in np.unique(molarr):
                    mol_to_indices[int(mid)] = np.where(molarr == mid)[0]
                used_source = "dump:mol"
        except Exception:
            mol_to_indices = {}
    if not mol_to_indices and getattr(u, "residues", None) is not None and len(u.residues) > 0:
        # Fallback: group by residue id
        for res in u.residues:
            mol_to_indices[int(res.resid)] = res.atoms.indices
        used_source = "residues"
    if not mol_to_indices:
        sys.stderr.write("Could not determine molecule groups: need mol IDs (molnums or dump 'mol') or residues.\n")
        sys.exit(2)

    # Per-atom properties
    masses = getattr(u.atoms, "masses", None)
    charges = getattr(u.atoms, "charges", None)
    # Try to fetch from dump columns if missing
    ts0 = u.trajectory[0]
    if (masses is None) or np.any(~np.isfinite(masses)):
        elem = None
        if hasattr(u.atoms, "elements") and u.atoms.elements is not None and np.all(u.atoms.elements != ""):
            elem = np.asarray(u.atoms.elements)
        elif hasattr(ts0, "data") and ts0.data is not None and "element" in ts0.data:
            elem = np.asarray(ts0.data["element"])
        if elem is not None:
            # Basic periodic table for common elements (amu)
            PT = {
                "H": 1.0079, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
                "Na": 22.990, "Mg": 24.305, "P": 30.974, "S": 32.06, "Cl": 35.45,
                "K": 39.098, "Ca": 40.078, "Zn": 65.38, "Br": 79.904, "I": 126.90,
            }
            masses = np.array([PT.get(str(e).strip(), 12.0) for e in elem], dtype=float)
        else:
            # Last resort: unit mass
            masses = np.ones(len(u.atoms), dtype=float)
            sys.stderr.write("Warning: masses missing; assuming unit masses for COM.\n")
    if (charges is None) or np.any(~np.isfinite(charges)):
        if hasattr(ts0, "data") and ts0.data is not None and "q" in ts0.data:
            charges = np.asarray(ts0.data["q"], dtype=float)
        else:
            sys.stderr.write("Atomic charges are required (missing both topology charges and dump 'q').\n")
            sys.exit(2)

    # Compute molecule net charge and total mass
    mol_charge: Dict[int, float] = {}
    mol_mass: Dict[int, float] = {}
    for mol_id, idx in mol_to_indices.items():
        mol_charge[mol_id] = float(np.sum(charges[idx]))
        mol_mass[mol_id] = float(np.sum(masses[idx]))

    species_map, neutral_species = identify_species(mol_charge)
    if not species_map and (neutral_species is None or len(neutral_species.mol_ids) == 0):
        sys.stderr.write("No charged species found; cannot compute ionic conductivity.\n")
        sys.exit(2)

    # Optionally split neutrals by residue name if available and requested
    neutral_groups = None
    neutral_group_masses = None
    if args.group_neutrals_by_resname and neutral_species is not None and getattr(u.atoms, 'resnames', None) is not None:
        neutral_groups = {}
        for mid in neutral_species.mol_ids:
            idx0 = int(mol_idx_arrays.get(mid, np.array([0]))[0])
            rn = str(u.atoms[idx0].resname)
            neutral_groups.setdefault(rn, []).append(mid)
        neutral_group_masses = {rn: float(np.mean([mol_mass[m] for m in mids])) for rn, mids in neutral_groups.items()}

    # Order species by charge magnitude/sign for stable output
    species_keys = sorted(species_map.keys(), key=lambda k: (abs(k[0]), k[0]))

    # Collect COM time series per molecule
    n_frames_total = len(u.trajectory)
    dt_ps = float(args.dt_ps)
    frames_sel = range(args.skip_frames, n_frames_total)
    n_frames = len(frames_sel)
    times_ps = np.arange(n_frames, dtype=float) * dt_ps
    com_series: Dict[int, np.ndarray] = {}  # mol_id -> (n_frames, 3)
    system_com = np.zeros((n_frames, 3), dtype=float)
    volumes = np.zeros(n_frames, dtype=float)

    # Pre-compute per-atom indices arrays for speed
    mol_idx_arrays = {mol_id: np.asarray(idx, dtype=int) for mol_id, idx in mol_to_indices.items()}

    for fi0, ts in enumerate(u.trajectory):
        if fi0 < args.skip_frames:
            continue
        fi = fi0 - args.skip_frames
        # Prefer unwrapped coordinates from dump if available
        pos = None
        if hasattr(ts, "data") and ts.data is not None:
            if all(k in ts.data for k in ("xu", "yu", "zu")):
                pos = np.column_stack([ts.data["xu"], ts.data["yu"], ts.data["zu"]])
            elif all(k in ts.data for k in ("x", "y", "z", "ix", "iy", "iz")):
                # reconstruct unwrapped from image flags: x + ix*Lx, etc.
                a, b, c = ts.dimensions[:3]
                pos = np.column_stack([
                    ts.data["x"] + ts.data["ix"] * a,
                    ts.data["y"] + ts.data["iy"] * b,
                    ts.data["z"] + ts.data["iz"] * c,
                ])
        if pos is None:
            pos = u.atoms.positions  # (n_atoms, 3) Angstrom
        volumes[fi] = _volume_from_dims(ts.dimensions)
        total_mass = float(np.sum(masses))
        system_com[fi, :] = np.einsum("i,ij->j", masses, pos) / total_mass
        for mol_id, idx in mol_idx_arrays.items():
            com_series.setdefault(mol_id, np.zeros((n_frames, 3), dtype=float))
            com_series[mol_id][fi, :] = compute_com(pos[idx], masses[idx])

    # Remove global translation by subtracting system COM at each frame
    for mol_id in com_series:
        com_series[mol_id] = com_series[mol_id] - system_com

    # Determine fitting window once (shared for self and Onsager)
    window_type = "frames"
    if args.fit_window_frames is not None:
        fit_i0, fit_i1 = args.fit_window_frames
    else:
        # If user left fractional window at default (0.5,0.9), use legacy default [50,200] frames
        if tuple(args.fit_window) == (0.5, 0.9):
            fit_i0, fit_i1 = 50, 200
            window_type = "frames_default"
        else:
            fit_i0 = int(np.floor(args.fit_window[0] * n_frames))
            fit_i1 = int(np.floor(args.fit_window[1] * n_frames))
            window_type = "frac"
    fit_i1 = max(fit_i1, fit_i0 + 2)  # ensure at least two points

    mol_slopes_A2_per_ps: Dict[int, float] = {}
    # Use time-averaged kMSD per molecule and per coordinate, mirroring md_utils
    lags = np.arange(n_frames, dtype=float)  # in frames
    for mol_id, xyz in com_series.items():
        msd_x = correlate_xy_np(xyz[:, 0], xyz[:, 0])
        msd_y = correlate_xy_np(xyz[:, 1], xyz[:, 1])
        msd_z = correlate_xy_np(xyz[:, 2], xyz[:, 2])
        msd_r = msd_x + msd_y + msd_z  # Angstrom^2 per lag
        slope_per_frame = linear_fit_slope(lags[fit_i0:fit_i1], msd_r[fit_i0:fit_i1])
        mol_slopes_A2_per_ps[mol_id] = slope_per_frame / dt_ps  # convert to A^2/ps

    # Aggregate diffusion per charged species only: D = slope / 6 (convert to 1e-10 m^2/s)
    D_species_1e10: Dict[int, float] = {}
    N_species: Dict[int, int] = {}
    species_masses: Dict[int, float] = {}
    for (qint, _sign), sp in species_map.items():
        slopes = [mol_slopes_A2_per_ps[mol_id] for mol_id in sp.mol_ids]
        D_A2_ps = np.mean(slopes) / 6.0
        D_1e10 = D_A2_ps * 100.0  # 1 A^2/ps = 100 × (1e-10 m^2/s)
        D_species_1e10[qint] = float(D_1e10)
        N_species[qint] = len(sp.mol_ids)
        # average molecular mass of species (amu)
        species_masses[qint] = float(np.mean([mol_mass[m] for m in sp.mol_ids]))

    V_avg_A3 = float(np.mean(volumes))
    L_box_A = V_avg_A3 ** (1.0 / 3.0)

    # Yeh–Hummer optional correction (applied to NE only)
    if args.viscosity_cp is not None:
        dD_1e10 = yeh_hummer_delta_D_1e10(args.temperature, args.viscosity_cp, L_box_A)
    else:
        dD_1e10 = 0.0
    D_species_inf_1e10 = {q: (D_species_1e10[q] + dD_1e10) for q in D_species_1e10}

    sigma_ne = nernst_einstein_conductivity(D_species_1e10, N_species, args.temperature, V_avg_A3)
    sigma_ne_yh = nernst_einstein_conductivity(D_species_inf_1e10, N_species, args.temperature, V_avg_A3)

    # Optional Onsager calculation
    sigma_onsager = None
    Lambda_1e10 = None
    Lambda_raw_1e10 = None
    Lambda_labels = None
    L_hat_SI = None
    if args.mode == "onsager":
        # Build species-summed COM series relative to system COM
        charged_keys_sorted = sorted(N_species.keys(), key=lambda q: (abs(q), q))
        # Build a list of keys including optional neutral groups with unique labels
        full_keys = [("charge", q) for q in charged_keys_sorted]
        if neutral_groups is not None:
            for rn in sorted(neutral_groups.keys()):
                full_keys.append(("neutral", rn))
        elif neutral_species is not None and len(neutral_species.mol_ids) > 0:
            full_keys.append(("neutral", "all"))
        nsp = len(full_keys)
        Rx = np.zeros((nsp, n_frames), dtype=float)
        Ry = np.zeros((nsp, n_frames), dtype=float)
        Rz = np.zeros((nsp, n_frames), dtype=float)
        for i, key in enumerate(full_keys):
            if key[0] == "charge":
                q = key[1]
                ids = species_map[(q, int(np.sign(q)))].mol_ids
            else:
                # neutral
                if neutral_groups is not None and key[1] in neutral_groups:
                    ids = neutral_groups[key[1]]
                else:
                    ids = neutral_species.mol_ids  # type: ignore[union-attr]
            for fi in range(n_frames):
                # Sum of molecule COMs at frame fi
                sx = 0.0
                sy = 0.0
                sz = 0.0
                for m in ids:
                    sx += com_series[m][fi, 0]
                    sy += com_series[m][fi, 1]
                    sz += com_series[m][fi, 2]
                # subtract N_i * system COM
                Rx[i, fi] = sx - len(ids) * system_com[fi, 0]
                Ry[i, fi] = sy - len(ids) * system_com[fi, 1]
                Rz[i, fi] = sz - len(ids) * system_com[fi, 2]

        # Compute kMSD cross-terms via FFT-based correlate
        lags = np.arange(n_frames, dtype=float)  # in frames
        # Reuse previously selected window
        w0, w1 = fit_i0, fit_i1
        w1 = max(w1, w0 + 2)
        kmsd_slope_A2_ps = np.zeros((nsp, nsp), dtype=float)
        for i in range(nsp):
            for j in range(i, nsp):
                msd_x = correlate_xy_np(Rx[i], Rx[j])
                msd_y = correlate_xy_np(Ry[i], Ry[j])
                msd_z = correlate_xy_np(Rz[i], Rz[j])
                msd_r = msd_x + msd_y + msd_z
                slope_per_frame = linear_fit_slope(lags[w0:w1], msd_r[w0:w1])
                val = slope_per_frame / dt_ps  # A^2/ps
                kmsd_slope_A2_ps[i, j] = val
                kmsd_slope_A2_ps[j, i] = val

        # Total number of molecules across charged and neutral species
        if neutral_groups is not None:
            N_total = float(sum(N_species.values()) + sum(len(v) for v in neutral_groups.values()))
        else:
            N_total = float(sum(N_species.values()) + (len(neutral_species.mol_ids) if neutral_species else 0))
        # Convert slopes to Lambda (1e-10 m^2/s units)
        Lambda_raw_1e10 = (kmsd_slope_A2_ps * 100.0) / (6.0 * N_total)
        # Remove COM error
        masses_vec_list = []
        for key in full_keys:
            if key[0] == "charge":
                masses_vec_list.append(species_masses[key[1]])
            else:
                if neutral_groups is not None and key[1] in neutral_group_masses:
                    masses_vec_list.append(neutral_group_masses[key[1]])
                else:
                    masses_vec_list.append(float(np.mean([mol_mass[m] for m in (neutral_species.mol_ids if neutral_species else [])])))
        masses_vec = np.array(masses_vec_list, dtype=float)
        Lambda_1e10 = remove_com_error(Lambda_raw_1e10, masses_vec)
        # Conductivity from Lambda
        q_vec = np.array([ (k[1] if k[0]=="charge" else 0.0) for k in full_keys], dtype=float)
        V_m3 = V_avg_A3 * 1e-30
        kB = 1.380_649e-23
        e = 1.602_176_634e-19
        Lambda_SI = Lambda_1e10 * 1e-10
        L_hat = Lambda_SI * (N_total / (V_m3 * kB * args.temperature))
        L_hat_SI = L_hat
        sigma_S_m = float(q_vec @ (L_hat @ q_vec) * (e**2))
        #sigma_onsager = sigma_S_m / 10.0
        sigma_onsager = sigma_S_m * 10.0
        # Build species axis labels aligned with Lambda rows/cols
        Lambda_labels = [
            (f"charge:{int(k[1])}" if k[0] == "charge" else f"neutral:{k[1]}")
            for k in full_keys
        ]
        # Optionally compute transference numbers per species (charged entries only)
        transference = None
        if args.compute_transference:
            denom = float(q_vec @ (L_hat @ q_vec))
            if denom != 0.0 and np.isfinite(denom):
                t_list = []
                for idx, lbl in enumerate(Lambda_labels):
                    q_i = q_vec[idx]
                    if q_i == 0.0:
                        continue
                    contrib = float(q_i * (L_hat @ q_vec)[idx])
                    t_i = contrib / denom
                    t_list.append((lbl, t_i))
                transference = t_list
            else:
                transference = []

    # Output
    out = {
        "method": ("Onsager" if args.mode == "onsager" else "Nernst-Einstein"),
        "temperature_K": float(args.temperature),
        "avg_volume_A3": V_avg_A3,
        "dt_ps": dt_ps,
        "n_frames": n_frames,
        "species": {
            str(q): {
                "count": int(N_species[q]),
                "mass_amu": float(species_masses[q]),
                "D_self_1e-10_m2_s": float(D_species_1e10[q]),
                "D_self_inf_1e-10_m2_s": float(D_species_inf_1e10[q]),
            }
            for q in sorted(D_species_1e10.keys(), key=lambda x: (abs(x), x))
        },
        "conductivity_NE_mS_per_cm": float(sigma_ne),
        "conductivity_NE_YH_mS_per_cm": float(sigma_ne_yh),
        "conductivity_Onsager_mS_per_cm": (None if sigma_onsager is None else float(sigma_onsager)),
        "neutral_molecule_count": int(len(neutral_species.mol_ids)) if neutral_species else 0,
        "fit_window_type": window_type,
        "fit_window_frames": [int(fit_i0), int(fit_i1)],
        "fit_window_frac": [float(fit_i0) / float(n_frames), float(fit_i1) / float(n_frames)],
        "yh_correction_applied": bool(args.viscosity_cp is not None and args.viscosity_cp > 0),
        "notes": "Species grouped by integer net charge per molecule; Onsager uses kMSD cross-correlations.",
    }

    # Conditionally include Lambda and/or L_hat matrices in output (Onsager mode)
    if args.mode == "onsager":
        if args.print_lambda and Lambda_1e10 is not None:
            out.update({
                "Lambda_unit": "10^-10 m^2/s",
                "Lambda_species_axis": (None if Lambda_labels is None else list(Lambda_labels)),
                "Lambda_raw": (None if Lambda_raw_1e10 is None else np.asarray(Lambda_raw_1e10).tolist()),
                "Lambda_com_removed": (None if Lambda_1e10 is None else np.asarray(Lambda_1e10).tolist()),
            })
        if args.print_L_hat and L_hat_SI is not None:
            out.update({
                "L_hat_unit": "1/(J s m)",
                "L_hat": np.asarray(L_hat_SI).tolist(),
                "L_hat_species_axis": (None if Lambda_labels is None else list(Lambda_labels)),
            })
        if args.compute_transference and transference is not None:
            # Provide as a mapping from label to value, and convenience keys for +/- if present
            trans_map = {lbl: float(val) for (lbl, val) in transference}
            out.update({
                "transference_numbers": trans_map,
            })
            # Convenience extraction for +1 and -1 using exact label match
            t_plus = trans_map.get("charge:1", None)
            t_minus = trans_map.get("charge:-1", None)
            out.update({
                "t_plus_charge_+1": (float(t_plus) if t_plus is not None else None),
                "t_minus_charge_-1": (float(t_minus) if t_minus is not None else None),
            })

    import json
    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
