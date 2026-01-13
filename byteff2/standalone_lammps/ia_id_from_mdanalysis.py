"""
Compute MD-based association (Ia) and domain network (Id) indexes from MD trajectories.

Definitions (from the paper's Eq. (1) and Eq. (2)):
    Ia = <P_+-> / N
    Id = <A_+-> / <P_+->
where <P_+-> is the time-averaged number of Li–anion pairs (contacts) and <A_+-> is the
(time-averaged) association count between those pairs; N is the number of cations (or anions),
so that Ia∈[0,1] and Id≥1 (Id grows with clustering). We follow Supplementary Note 1's
coordination rule: a Li+ and an anion are considered paired if any O/N/F atom of that anion
lies within 2.8 Å of the Li+.

High-level algorithm per frame
------------------------------
1) Find all Li–anion contact pairs using a cutoff (default 2.8 Å) between Li atoms and the
   set of anion atoms eligible for contact (e.g., O/N/F in FSI−).
2) Define a *pair* as a unique tuple (li_index, anion_molid).
3) Build a pair–pair association graph where two pairs are *associated* if they share
   the same Li or the same anion molecule (i.e., overlapping coordination). 
   In the absence of aggregation, each pair has only itself; we count one intrinsic
   self-association per pair so that A = P when there is no clustering.
4) For the frame, compute:
       P = number of pairs (nodes)
       E = number of undirected edges in the association graph (connections between distinct pairs)
       A = P + 2*E   (because sum_{nodes} (1 + degree) = P + 2E)
5) Average P and A over the chosen trajectory window to get <P> and <A>, then compute Ia, Id.

Notes
-----
* N is set to min(n_cations, n_anions) so Ia ≤ 1 even if stoichiometry is imbalanced.
* Periodic boundary conditions are honored using the box from the trajectory.
* Works with LAMMPS data/dump when molecule IDs are present (mapped to MDAnalysis "resids").
* You must provide three selections:
    - cation_sel: selects Li+ atoms (e.g., "type 1" or "name Li")
    - anion_sel_all: selects ALL atoms that belong to the anion molecules, used to count anions (e.g., "resname FSI")
    - anion_contact_sel: selects the subset of anion atoms eligible for contact (e.g., O/N/F: "resname FSI and (name O* or name N* or name F*)" or by types)

CLI examples
------------
Python:
    python Ia_Id_from_MDAnalysis.py \
        --top electrolyte.data --traj traj.lammpstrj \
        --cation-sel "type 1" \
        --anion-sel-all "resname FSI" \
        --anion-contact-sel "resname FSI and (name O* or name N* or name F*)" \
        --cutoff 2.8 --start 0 --stop -1 --step 1 --out metrics.csv

Jupyter/Script API:
    from Ia_Id_from_MDAnalysis import compute_Ia_Id
    Ia, Id, summary = compute_Ia_Id(u, cation_sel, anion_sel_all, anion_contact_sel)

Outputs
-------
* Prints Ia, Id, and CN = Ia*Id.
* Optional CSV with per-frame metrics and cumulative averages for convergence checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, FrozenSet

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance


@dataclass
class FrameMetrics:
    frame: int
    time_ps: Optional[float]
    P_pairs: int
    E_edges: int
    A_assoc: int
    Ia_inst: float
    Id_inst: float
    Ia_cumavg: float
    Id_cumavg: float


def _unique_resids(ag: mda.core.groups.AtomGroup) -> np.ndarray:
    """Return the unique residue IDs for an AtomGroup.

    For LAMMPS, MDAnalysis normally maps molecule-ids to `resids`. If resids are all zeros
    and no molecules are defined, this function will still return [0] to avoid crashes, but
    proper anion counting requires molecule/residue information in the topology.
    """
    resids = ag.resids
    if resids is None or len(resids) == 0:
        return np.array([], dtype=int)
    return np.unique(resids)


def _build_pairs_and_edges(
    u: mda.Universe,
    li_ag: mda.core.groups.AtomGroup,
    anion_contact_ag: mda.core.groups.AtomGroup,
    cutoff: float,
) -> Tuple[int, int, int]:
    """Return (P_pairs, E_edges, U_cations) for the current frame.

    P = number of unique (Li_index, anion_resid) contact pairs.
    E = number of undirected edges between pairs that *share* the same Li or the same anion.
    """
    # Compute contacts between Li atoms and anion-contact atoms (O/N/F, etc.), PBC-aware
    box = u.trajectory.ts.dimensions  # [lx, ly, lz, alpha, beta, gamma]
    # Map from anion-contact local index to global Atom and resid
    anion_atoms = anion_contact_ag.atoms
    li_atoms = li_ag.atoms

    if li_atoms.n_atoms == 0 or anion_atoms.n_atoms == 0:
        return 0, 0

    # Pair search; returns indices into the *provided* arrays
    # Robustly handle different MDAnalysis return signatures (2-tuple or 3-tuple)
    try:
        out_tuple = capped_distance(
            li_atoms.positions, anion_atoms.positions, max_cutoff=cutoff, box=box
        )
        if isinstance(out_tuple, tuple) and len(out_tuple) == 3:
            idx_li, idx_anion, _ = out_tuple
        else:
            idx_li, idx_anion = out_tuple
        # Validate that returned indices are within the local AtomGroup bounds; some
        # MDAnalysis versions may return universe-level indices. If we detect any
        # out-of-range index, fall back to a safe distance-matrix method.
        if (
            len(idx_li) > 0 and (np.max(idx_li) >= li_atoms.n_atoms or np.min(idx_li) < 0)
        ) or (
            len(idx_anion) > 0 and (np.max(idx_anion) >= anion_atoms.n_atoms or np.min(idx_anion) < 0)
        ):
            raise IndexError("capped_distance returned global indices; falling back")
    except Exception:
        # Safe fallback: compute full distance matrix and threshold
        from MDAnalysis.lib.distances import distance_array
        D = distance_array(li_atoms.positions, anion_atoms.positions, box=box)
        idx_li, idx_anion = np.where(D <= cutoff)

    # Normalize indices to flat 1D arrays (handles object arrays / ragged outputs)
    idx_li = np.asarray(idx_li).ravel()
    idx_anion = np.asarray(idx_anion).ravel()
    n_pairs = int(min(idx_li.size, idx_anion.size))

    # Build unique pairs (Li index, anion resid)
    pair_nodes: Dict[Tuple[int, int], int] = {}
    by_li: Dict[int, List[int]] = defaultdict(list)  # Li_index -> list of node ids
    by_anion: Dict[int, List[int]] = defaultdict(list)  # anion_resid -> list of node ids

    node_counter = 0
    for k in range(n_pairs):
        i_li = int(idx_li[k])
        j_an = int(idx_anion[k])
        if i_li < 0 or i_li >= li_atoms.n_atoms or j_an < 0 or j_an >= anion_atoms.n_atoms:
            continue  # guard against any stray indices
        li_idx = int(li_atoms[i_li].index)
        an_resid = int(anion_atoms[j_an].resid)
        key = (li_idx, an_resid)
        if key not in pair_nodes:
            node_id = node_counter
            pair_nodes[key] = node_id
            node_counter += 1
            by_li[li_idx].append(node_id)
            by_anion[an_resid].append(node_id)

    P = len(pair_nodes)
    # Count unique Li cations that participate in at least one pair (works also when P<=1)
    unique_li_in_pairs = set(li_idx for (li_idx, _an_resid) in pair_nodes.keys())
    U = len(unique_li_in_pairs)
    if P <= 1:
        # No edges possible; return a 3-tuple consistently
        return P, 0, U

    # Undirected edges between pairs that share the same Li or the same anion
    edges: set[frozenset[int]] = set()

    def add_edges_from_groups(d: Dict[int, List[int]]):
        for _, nodes in d.items():
            if len(nodes) > 1:
                for a, b in combinations(nodes, 2):
                    if a != b:
                        edges.add(frozenset((a, b)))

    add_edges_from_groups(by_li)
    add_edges_from_groups(by_anion)

    E = len(edges)
    return P, E, U


@dataclass
class IaIdResult:
    Ia: float
    Id: float
    CN: float
    P_avg: float
    A_avg: float
    N_ref: int


def compute_Ia_Id(
    u: mda.Universe,
    cation_sel: str,
    anion_sel_all: str,
    anion_contact_sel: str,
    cutoff: float = 2.8,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: int = 1,
    record_timeseries: bool = True,
) -> Tuple[float, float, IaIdResult | Dict[str, np.ndarray]]:
    """Compute Ia and Id from a trajectory window.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe with topology and trajectory loaded.
    cation_sel : str
        Selection string for Li+ atoms (e.g., "type 1" or "name Li").
    anion_sel_all : str
        Selection for *all* atoms in anion molecules (e.g., "resname FSI"). Used to count anions.
    anion_contact_sel : str
        Selection for contact-eligible anion atoms (subset of anion_sel_all), e.g., O/N/F atoms.
    cutoff : float
        Distance cutoff in Å for Li–anion contact (default 2.8 Å).
    start, stop, step : int
        Frame window (Python slicing semantics). Defaults: full trajectory.
    record_timeseries : bool
        If True, returns per-frame metrics arrays in the details dict.

    Returns
    -------
    Ia : float
        Association index.
    Id : float
        Domain network index.
    details : IaIdResult or dict
        If record_timeseries=False: an IaIdResult summary object.
        Else: a dict containing arrays for convergence analysis plus the summary in keys.
    """
    li_ag = u.select_atoms(cation_sel)
    an_all_ag = u.select_atoms(anion_sel_all)
    an_contact_ag = u.select_atoms(anion_contact_sel)

    if li_ag.n_atoms == 0:
        raise ValueError("Cation selection returned 0 atoms. Check `cation_sel`.")
    if an_all_ag.n_atoms == 0:
        raise ValueError("Anion (all) selection returned 0 atoms. Check `anion_sel_all`.")
    if an_contact_ag.n_atoms == 0:
        raise ValueError("Anion contact selection returned 0 atoms. Check `anion_contact_sel`.")

    # N reference = min(# Li atoms, # anion molecules)
    n_cations = int(li_ag.n_atoms)
    unique_anion_resids = _unique_resids(an_all_ag)
    if unique_anion_resids.size == 0:
        raise ValueError(
            "Could not determine anion molecule IDs (resids are empty). Make sure your topology carries molecule IDs."
        )
    n_anions = int(unique_anion_resids.size)
    # Normalize Ia by the number of cations so Ia∈[0,1]
    N_ref = n_cations

    # Iterate trajectory window
    traj = u.trajectory
    n_frames_total = len(traj)
    i0 = 0 if start is None else (start if start >= 0 else n_frames_total + start)
    i1 = n_frames_total if stop is None or stop == -1 else (stop if stop >= 0 else n_frames_total + stop)

    P_list: List[int] = []
    U_list: List[int] = []
    A_list: List[int] = []
    E_list: List[int] = []
    t_list: List[Optional[float]] = []

    for ts in traj[i0:i1:step]:
        P, E, U = _build_pairs_and_edges(u, li_ag, an_contact_ag, cutoff)
        A = P + 2 * E  # intrinsic self (+1 per pair) plus 2 per undirected edge

        P_list.append(P)
        U_list.append(U)
        E_list.append(E)
        A_list.append(A)
        # time in ps if available; MDAnalysis sets None if not present
        t_list.append(getattr(ts, "time", None))

    P_arr = np.asarray(P_list, dtype=float)
    U_arr = np.asarray(U_list, dtype=float)
    A_arr = np.asarray(A_list, dtype=float)

    # Averages and indexes
    P_avg = float(P_arr.mean()) if P_arr.size else 0.0
    U_avg = float(U_arr.mean()) if U_arr.size else 0.0
    A_avg = float(A_arr.mean()) if A_arr.size else 0.0

    Ia = (U_avg / N_ref) if N_ref > 0 else 0.0
    Id = (A_avg / P_avg) if P_avg > 0 else float("nan")
    CN = Ia * Id  # optional: reported in the SI tables as CN = Ia*Id

    summary = IaIdResult(Ia=Ia, Id=Id, CN=CN, P_avg=P_avg, A_avg=A_avg, N_ref=N_ref)

    if not record_timeseries:
        return Ia, Id, summary

    # Per-frame instantaneous and cumulative averages for convergence checks
    Ia_inst = U_arr / N_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        Id_inst = np.where(P_arr > 0, A_arr / P_arr, np.nan)

    Ia_cumavg = np.array([np.nanmean(Ia_inst[:k]) for k in range(1, Ia_inst.size + 1)])
    Id_cumavg = np.array([np.nanmean(Id_inst[:k]) for k in range(1, Id_inst.size + 1)])

    details = {
        "time_ps": np.array(t_list),
        "P_pairs": P_arr,
        "U_cations": U_arr,
        "E_edges": np.array(E_list, dtype=float),
        "A_assoc": A_arr,
        "Ia_inst": Ia_inst,
        "Id_inst": Id_inst,
        "Ia_cumavg": Ia_cumavg,
        "Id_cumavg": Id_cumavg,
        "Ia": Ia,
        "Id": Id,
        "CN": CN,
        "P_avg": P_avg,
        "U_avg": U_avg,
        "A_avg": A_avg,
        "N_ref": N_ref,
        "n_frames": P_arr.size,
        "start": i0,
        "stop": i1,
        "step": step,
        "cutoff_A": cutoff,
        "cation_sel": cation_sel,
        "anion_sel_all": anion_sel_all,
        "anion_contact_sel": anion_contact_sel,
    }
    return Ia, Id, details


def _write_csv(details: Dict[str, np.ndarray], out_csv: str):
    import csv

    fields = [
        "frame",
        "time_ps",
        "P_pairs",
        "E_edges",
        "A_assoc",
        "Ia_inst",
        "Id_inst",
        "Ia_cumavg",
        "Id_cumavg",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Ia", details["Ia"], "Id", details["Id"], "CN", details["CN"], "N_ref", details["N_ref"]])
        w.writerow(fields)
        for k in range(details["n_frames"]):
            row = [
                details["start"] + k * details["step"],
                (None if details["time_ps"][k] is None else float(details["time_ps"][k])),
                int(details["P_pairs"][k]),
                int(details["E_edges"][k]),
                int(details["A_assoc"][k]),
                float(details["Ia_inst"][k]),
                float(details["Id_inst"][k]) if np.isfinite(details["Id_inst"][k]) else "nan",
                float(details["Ia_cumavg"][k]),
                float(details["Id_cumavg"][k]) if np.isfinite(details["Id_cumavg"][k]) else "nan",
            ]
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Compute Ia and Id from MD trajectories using MDAnalysis.")
    p.add_argument("--top", required=True, help="Topology file (e.g., LAMMPS data, PSF, PDB)")
    p.add_argument("--traj", required=True, nargs="+", help="Trajectory file(s) (e.g., LAMMPS dump)")
    p.add_argument("--cation-sel", required=True, help="MDAnalysis selection for Li+ atoms")
    p.add_argument("--anion-sel-all", required=True, help="Selection for all atoms belonging to anions")
    p.add_argument(
        "--anion-contact-sel",
        required=True,
        help="Selection for contact-eligible anion atoms (subset of --anion-sel-all)",
    )
    p.add_argument("--cutoff", type=float, default=2.8, help="Contact cutoff in Å (default: 2.8 Å)")
    p.add_argument("--start", type=int, default=None, help="Start frame (default: 0)")
    p.add_argument("--stop", type=int, default=-1, help="Stop frame (exclusive; -1 = end)")
    p.add_argument("--step", type=int, default=1, help="Stride in frames (default: 1)")
    p.add_argument("--out", default=None, help="Optional CSV path to write per-frame metrics")

    args = p.parse_args()

    u = mda.Universe(args.top, *args.traj)
    Ia, Id, details = compute_Ia_Id(
        u,
        cation_sel=args.cation_sel,
        anion_sel_all=args.anion_sel_all,
        anion_contact_sel=args.anion_contact_sel,
        cutoff=args.cutoff,
        start=args.start,
        stop=args.stop,
        step=args.step,
        record_timeseries=True,
    )

    print(f"Ia = {Ia:.4f}")
    print(f"Id = {Id:.4f}")
    print(f"CN = Ia*Id = {details['CN']:.4f}")
    print(f"<U> (unique Li paired) = {details['U_avg']:.3f}, <P> (pairs) = {details['P_avg']:.3f}, <A> = {details['A_avg']:.3f}, N_cations = {details['N_ref']}")

    if args.out:
        _write_csv(details, args.out)
        print(f"Per-frame metrics written to: {args.out}")


if __name__ == "__main__":
    main()
