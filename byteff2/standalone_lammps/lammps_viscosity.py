#!/usr/bin/env python3
"""
Compute shear viscosity from LAMMPS stress tensor time series via Green–Kubo.

Method: Green–Kubo integral of the off-diagonal stress autocorrelation function
  eta = V / (k_B T) * ∫_0^∞ dt < P_xy(0) P_xy(t) + P_xz(0) P_xz(t) + P_yz(0) P_yz(t) > / 3

Inputs
- A text/CSV file with time and off-diagonal stress components. Columns can be
  auto-detected or specified. Supports whitespace or comma-separated formats.
- Temperature in K.
- Volume as a constant (A^3) or derived from a trajectory via MDAnalysis.

Units
- Time column in ps (or use --time-units to convert from fs).
- Stress in Pa, atm, or bar (declare via --pressure-units). Internally converted to Pa.
- Viscosity is reported in cP (mPa·s).

Examples
  python lammps_viscosity.py --stress thermo.txt --temperature 298 \
    --pressure-units atm --time-units ps --volume 1.234e5

  # Use topology+trajectory to compute average volume instead of --volume
  python lammps_viscosity.py --stress stress.csv --temperature 298 \
    --pressure-units bar --data system.data --dump traj.dump
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _maybe_mdanalysis_available() -> bool:
    try:
        import MDAnalysis as mda  # noqa: F401
        return True
    except Exception:
        return False


def _volume_from_dims(dimensions: np.ndarray) -> float:
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
    return float(vol)


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """FFT-based autocorrelation of a zero-mean 1D series.

    Returns unbiased estimator (normalized by N - lag).
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    # Unbiased normalization
    ac /= (np.arange(n, 0, -1))
    return ac


def _integrate_trapz(y: np.ndarray, dt: float, tmax: Optional[float] = None) -> float:
    if tmax is not None:
        nmax = min(len(y), int(np.floor(tmax / dt)) + 1)
        y = y[:nmax]
    return float(np.trapz(y, dx=dt))


def _to_pa(values: np.ndarray, units: str) -> np.ndarray:
    units = units.lower()
    if units == "pa":
        return values
    if units == "atm":
        return values * 101325.0
    if units == "bar":
        return values * 1.0e5
    raise ValueError("Unsupported pressure units: use pa, atm, or bar")


def _to_ps(values: np.ndarray, units: str) -> np.ndarray:
    units = units.lower()
    if units == "ps":
        return values
    if units == "fs":
        return values * 1e-3
    if units == "ns":
        return values * 1e3
    raise ValueError("Unsupported time units: use fs, ps, or ns")


def _read_stress_table(
    path: str,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read time and off-diagonal stress components.

    Returns (time, Pxy, Pxz, Pyz) as numpy arrays.
    Auto-detects separators and tries common column names if not provided.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback to whitespace
        df = pd.read_csv(path, delim_whitespace=True, engine="python")

    if columns is None:
        # Heuristics: look for typical names/cases
        candidates = [
            ("time", ["time", "Time", "t", "Step", "step"]),
            ("pxy", ["Pxy", "pxy", "PXY", "p_xy", "sxy", "Sxy", "xy"]),
            ("pxz", ["Pxz", "pxz", "PXZ", "p_xz", "sxz", "Sxz", "xz"]),
            ("pyz", ["Pyz", "pyz", "PYZ", "p_yz", "syz", "Syz", "yz"]),
        ]
        mapping = {}
        for key, names in candidates:
            for nm in names:
                if nm in df.columns:
                    mapping[key] = nm
                    break
        missing = [k for k in ("time", "pxy", "pxz", "pyz") if k not in mapping]
        if missing:
            raise ValueError(
                "Could not auto-detect columns for: " + ", ".join(missing) +
                ". Provide --columns time,pxy,pxz,pyz or rename columns."
            )
        cols = [mapping["time"], mapping["pxy"], mapping["pxz"], mapping["pyz"]]
    else:
        if len(columns) != 4:
            raise ValueError("--columns must specify exactly 4 names: time,pxy,pxz,pyz")
        cols = list(columns)

    arr = df[cols].to_numpy(dtype=float)
    t, pxy, pxz, pyz = arr.T
    return t, pxy, pxz, pyz


def main():
    ap = argparse.ArgumentParser(description="Viscosity (Green–Kubo) from LAMMPS stress time series")
    ap.add_argument("--stress", required=True, help="Path to stress table (time and Pxy, Pxz, Pyz)")
    ap.add_argument("--columns", default=None, help="Column names as comma list: time,pxy,pxz,pyz")
    ap.add_argument("--pressure-units", default="atm", choices=["pa", "atm", "bar"],
                    help="Units of stress columns (default: atm)")
    ap.add_argument("--time-units", default="ps", choices=["fs", "ps", "ns"],
                    help="Units of time column (default: ps)")
    ap.add_argument("--temperature", type=float, required=True, help="Temperature in Kelvin")
    vol_group = ap.add_mutually_exclusive_group(required=True)
    vol_group.add_argument("--volume", type=float, help="Constant volume in Angstrom^3")
    vol_group.add_argument("--data", help="LAMMPS data file to get volume from trajectory")
    ap.add_argument("--dump", help="LAMMPS dump trajectory (required if --data is set)")
    ap.add_argument("--tmax-ps", type=float, default=None, help="Optional upper limit for GK integral (ps)")
    ap.add_argument("--json-out", default=None, help="Optional JSON output path")

    args = ap.parse_args()
    columns = None if args.columns is None else [c.strip() for c in args.columns.split(",")]

    # Read stress time series
    try:
        t_raw, pxy_raw, pxz_raw, pyz_raw = _read_stress_table(args.stress, columns)
    except Exception as exc:  # pylint: disable=broad-except
        sys.stderr.write(f"Failed to read stress file: {exc}\n")
        sys.exit(2)

    t_ps = _to_ps(t_raw, args.time_units)
    dt_ps = float(np.mean(np.diff(t_ps)))
    if not np.isfinite(dt_ps) or dt_ps <= 0:
        sys.stderr.write("Invalid or non-uniform time spacing detected; ensure time column is sorted and consistent.\n")
        sys.exit(2)

    # Volume
    if args.volume is not None:
        V_A3 = float(args.volume)
    else:
        if not args.dump:
            sys.stderr.write("--dump is required when using --data for volume estimation.\n")
            sys.exit(2)
        if not _maybe_mdanalysis_available():
            sys.stderr.write("MDAnalysis is required to read trajectory for volume. Install via: pip install MDAnalysis\n")
            sys.exit(2)
        import MDAnalysis as mda
        u = mda.Universe(args.data, args.dump, format=None)
        vols = []
        for ts in u.trajectory:
            vols.append(_volume_from_dims(ts.dimensions))
        if not vols:
            sys.stderr.write("Failed to read any frames for volume from trajectory.\n")
            sys.exit(2)
        V_A3 = float(np.mean(vols))

    # Convert stresses to Pa
    pxy = _to_pa(pxy_raw, args.pressure_units)
    pxz = _to_pa(pxz_raw, args.pressure_units)
    pyz = _to_pa(pyz_raw, args.pressure_units)

    # Autocorrelation for each component
    ac_xy = _autocorr_fft(pxy)
    ac_xz = _autocorr_fft(pxz)
    ac_yz = _autocorr_fft(pyz)
    ac_avg = (ac_xy + ac_xz + ac_yz) / 3.0

    # Green–Kubo integral
    kB = 1.380_649e-23  # J/K
    V_m3 = V_A3 * 1e-30
    dt_s = dt_ps * 1e-12
    integral = _integrate_trapz(ac_avg, dt_s, args.tmax_ps * 1e-12 if args.tmax_ps else None)
    eta_Pa_s = (V_m3 / (kB * args.temperature)) * integral
    eta_cP = eta_Pa_s * 1000.0  # 1 Pa·s = 1000 cP

    result = {
        "method": "Green-Kubo",
        "temperature_K": float(args.temperature),
        "avg_volume_A3": V_A3,
        "dt_ps": dt_ps,
        "n_samples": int(len(t_ps)),
        "tmax_ps": float(args.tmax_ps) if args.tmax_ps else None,
        "viscosity_cP": float(eta_cP),
        "notes": "Autocorrelation averaged over Pxy, Pxz, Pyz; unbiased FFT-based estimator.",
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

