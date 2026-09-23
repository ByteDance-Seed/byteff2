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
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Geometry.rdGeometry import Point3D


logger = logging.getLogger(__name__)

##########################################
##           conformers
##########################################


def get_conf_rms(mol: Chem.Mol, idx1: int, idx2: int) -> float:
    rms = AllChem.GetConformerRMS(mol, idx1, idx2, prealigned=False)
    return rms


def get_rms_matrix(mol: Chem.Mol) -> List:
    """rmsmatrix = [ a,
                        b, c,
                        d, e, f,
                        g, h, i, j]
    Where a is the RMS between conformers 0 and 1, b is the RMS between conformers 0 and 2, etc.
    This way it can be directly used as distance matrix in e.g. Butina clustering.
    """

    rms_m = AllChem.GetConformerRMSMatrix(mol, prealigned=False)

    return rms_m


def align_all_conformers(mol: Chem.Mol) -> List:
    """rmslist contains the RMS values between the first conformer and all others."""

    rmslist = []
    AllChem.AlignMolConformers(mol, RMSlist=rmslist)

    return rmslist


def find_hypervalent_centers(mol: Chem.Mol) -> List[dict]:
    """Detect hypervalent centers (non-carbon atoms with degree >= 5).

    Returns a list of dicts:
      {idx, symbol, degree, charge, neighbors}
    """

    res = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        atomic_num = atom.GetAtomicNum()
        if deg >= 5 and atomic_num in [15, 16]:
            res.append(
                {
                    "idx": atom.GetIdx(),
                    "symbol": atom.GetSymbol(),
                    "degree": deg,
                    "charge": atom.GetFormalCharge(),
                    # Directly attached atoms to the center (RDKit atom indices).
                    # NOTE: we use "neighbors" instead of "ligands" to avoid confusion with
                    # whole-molecule ligands in typical ligand/protein workflows.
                    "neighbors": [nb.GetIdx() for nb in atom.GetNeighbors()],
                }
            )
    return res


def _fix_hypervalent_bounds(mol_h: Chem.Mol, bm, hyper_atoms: List[dict]):
    """Relax neighbor-neighbor (1,3) distance bounds around hypervalent centers.

    RDKit distance-geometry embedding can fail when the bounds matrix is over-constrained.
    Here we only relax constraints between atoms attached to the same hypervalent center.

    Terminology:
      - "cis"  means two attached atoms are adjacent around the same center (ideal angle ~ 90°).
      - "trans" means two attached atoms are opposite around the same center (ideal angle ~ 180°).
    For a typical center-neighbor bond length r, the neighbor-neighbor distances are roughly:
      - cis  : ~ sqrt(2) * r
      - trans: ~ 2 * r
    We use a loose range to keep embedding solvable.
    """

    for info in hyper_atoms:
        center = info["idx"]
        neighbors = info["neighbors"]
        if len(neighbors) < 2:
            continue

        # Average center-neighbor bond length, estimated from bounds matrix (mean of upper/lower bounds).
        avg_bond = 0.0
        for ni in neighbors:
            a, b = min(center, ni), max(center, ni)
            avg_bond += (bm[a][b] + bm[b][a]) / 2
        avg_bond /= len(neighbors)

        # Relax neighbor-neighbor distance bounds to cover both cis (~sqrt(2)*bond)
        # and trans (~2*bond) arrangements with a loose, solvable range.
        lower_new = avg_bond * 1.2
        upper_new = avg_bond * 2.2

        for i, ni in enumerate(neighbors):
            for nj in neighbors[i + 1 :]:
                a, b = min(ni, nj), max(ni, nj)
                # RDKit bounds matrix convention: bm[b][a] is the lower bound, bm[a][b] is the upper bound.
                # Relaxing the interval means lowering the lower bound and raising the upper bound.
                bm[b][a] = min(bm[b][a], lower_new)
                bm[a][b] = max(bm[a][b], upper_new)

    return bm


def _embed_hypervalent_conformer(
    mol_h: Chem.Mol, *, hyper_atoms: List[dict], seed: int, n_threads: int
) -> Optional[np.ndarray]:
    """Generate one 3D conformer for hypervalent molecules.

    This keeps the fallback simple:
      1) relax bounds between neighbors around hypervalent centers
      2) embed with RDKit distance geometry

    Downstream force-field optimization (MMFF/UFF) is responsible for final relaxation.
    """

    mol = Chem.Mol(mol_h)
    mol.RemoveAllConformers()

    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    bm_relaxed = _fix_hypervalent_bounds(mol, bm, hyper_atoms)

    params = rdDistGeom.EmbedParameters()
    params.useRandomCoords = True
    params.randomSeed = int(seed)
    params.maxIterations = 5000
    params.numThreads = n_threads
    params.SetBoundsMat(bm_relaxed)

    ret = rdDistGeom.EmbedMolecule(mol, params)
    if ret != 0 or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=float)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i] = (p.x, p.y, p.z)

    return coords


def opt_confs(mol: Chem.Mol, *, ffoptimizer: str = "mmff94s", n_threads: int = 1, verbose: bool = False) -> List:
    """optimization conformers and calculated energies.

    Args:
    mol: rdkit mol
    ffoptimizer: the force field used to optimize conformers.
    n_threads: `0` means use all threads.
    is_verbose: used to debug.

    Returns:
    energies: the force field opt energies.
    """

    ffoptimizer = ffoptimizer.lower()
    ffoptimizers = ["mmff94", "mmff94s", "uff"]
    if ffoptimizer not in ffoptimizers:
        raise NotImplementedError(f"{ffoptimizer} is not avaliable, choose one from {ffoptimizers}")

    results = None
    ffopt_max_cycles = 20000
    hyper_atoms = find_hypervalent_centers(mol)

    n_confs = mol.GetNumConformers()
    if verbose:
        logger.info(f"There are {n_confs} conformers.")

    if ffoptimizer in ["mmff94", "mmff94s"]:
        # MMFF is generally better for organic molecules, but it does not cover many
        # hypervalent/charged species. For those hypervalent cases, keep the embedded
        # geometry instead of handing it to UFF, which may distort the structure.
        has_mmff_params = AllChem.MMFFHasAllMoleculeParams(mol)
        if has_mmff_params:
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol,
                mmffVariant=ffoptimizer,
                numThreads=n_threads,
                maxIters=ffopt_max_cycles,
                ignoreInterfragInteractions=False,
            )
        elif hyper_atoms:
            if verbose:
                logger.info(
                    "MMFF parameters missing for hypervalent centers %s; "
                    "keep embedded conformers and skip FF optimization.",
                    [(h["symbol"], h["idx"], h["degree"]) for h in hyper_atoms],
                )
            return [(0, 0.0) for _ in range(n_confs)]
        else:
            if verbose:
                logger.info("MMFF parameters missing; falling back to UFF for conformer optimization.")
            ffoptimizer = "uff"

    if ffoptimizer == "uff":
        results = AllChem.UFFOptimizeMoleculeConfs(
            mol,
            numThreads=n_threads,
            maxIters=ffopt_max_cycles,
            ignoreInterfragInteractions=False,
        )

    # gather energies in hartree and check results
    # results is a list of (not_converged, energy) 2-tuples. energy in kcal/mol
    # If not_converged is 0 the optimization converged for that conformer.
    assert results is not None, f"OptimizeMoleculeConfs failed with {ffoptimizer}"

    return results


def generate_confs(
    rkmol: Chem.Mol, *, nconfs: int = 1, ffopt: bool = True, n_threads: int = 1, verbose: bool = False, **kwargs
) -> Tuple:
    """generate conformers for rkmol

    REQUIREMENT: rkmol must have been sanitized, otherwise an error is raised

    Args:
    rkmol:
    nconfs: The target number of conformers to generate.

    Returns:
    elements: the elements of atoms in mol.
    coordinates: all conformers coordination.
    energies: opt energies.
    mol:

    Raises:
    ValueError:
        1. Embed molecules failed.
        2. Opt failed.
    """

    # embed conformers settings
    # https://www.rdkit.org/docs/RDKit_Book.html?highlight=etversion#parameters-controlling-conformer-generation
    params = AllChem.EmbedParameters()
    params.ETversion = 2  # for both ETKDGv2 and ETKDGv3 this should be 2
    params.useSymmetryForPruning = kwargs.pop("useSymmetryForPruning", True)
    params.useBasicKnowledge = kwargs.pop("useBasicKnowledge", True)
    params.enforceChirality = kwargs.pop("enforceChirality", True)

    params.useSmallRingTorsions = kwargs.pop("useSmallRingTorsions", False)
    params.useMacrocycleTorsions = kwargs.pop("useMacrocycleTorsions", False)
    params.useExpTorsionAnglePrefs = kwargs.pop("useExpTorsionAnglePrefs", False)

    params.useRandomCoords = kwargs.pop("useRandomCoords", False)  # useful for large molecules
    params.randomSeed = kwargs.pop("randomSeed", 42)
    params.pruneRmsThresh = kwargs.pop("pruneRmsThresh", -1.0)  # no prune by default
    params.maxIterations = kwargs.pop("maxIterations", 1000)
    params.numThreads = n_threads

    params.clearConfs = True
    params.embedFragmentsSeparately = True
    params.forceTransAmides = True

    n_confs = nconfs * kwargs.pop("nconfs_multiplier", 1)

    conf_ids = AllChem.EmbedMultipleConfs(rkmol, n_confs, params)
    if not conf_ids and not params.useRandomCoords:
        params.useRandomCoords = True
        if verbose:
            logger.info("Retry EmbedMultipleConfs using random_coord=True")
        conf_ids = AllChem.EmbedMultipleConfs(rkmol, n_confs, params)

    if not conf_ids:
        # For molecules with hypervalent centers (degree >= 5), RDKit DG embedding may
        # systematically fail. Fall back to a "relaxed bounds + local geometry fix" strategy.
        hyper_atoms = find_hypervalent_centers(rkmol)
        if not hyper_atoms:
            raise ValueError("EmbedMultipleConfs failed.")

        if verbose:
            logger.info(
                "EmbedMultipleConfs failed; detected hypervalent centers %s, trying hypervalent embedding fallback.",
                [(h["symbol"], h["idx"], h["degree"]) for h in hyper_atoms],
            )

        # Clear existing conformers and generate the target number with the custom workflow.
        rkmol.RemoveAllConformers()
        natoms = rkmol.GetNumAtoms(onlyExplicit=False)

        # Use a deterministic seed for reproducibility; each attempt uses seed + attempt.
        base_seed = params.randomSeed if params.randomSeed not in (-1, 0) else 42
        max_attempts = kwargs.pop("hypervalent_max_attempts", max(30, n_confs * 30))

        conf_ids = []
        attempt = 0
        while len(conf_ids) < n_confs and attempt < int(max_attempts):
            coords = _embed_hypervalent_conformer(
                rkmol,
                hyper_atoms=hyper_atoms,
                seed=base_seed + attempt,
                n_threads=n_threads,
            )
            attempt += 1
            if coords is None:
                continue
            rkconf = Chem.Conformer(natoms)
            for atom_idx in range(natoms):
                rkconf.SetAtomPosition(atom_idx, Point3D(*coords[atom_idx].tolist()))
            conf_ids.append(rkmol.AddConformer(rkconf, assignId=True))

        if not conf_ids:
            raise ValueError(
                "EmbedMultipleConfs failed for hypervalent molecule; hypervalent fallback embedding also failed."
            )
    if verbose:
        logger.info(f"{len(conf_ids)}/{n_confs} conformers are generated.")

    if ffopt:
        results = opt_confs(rkmol, ffoptimizer="mmff94s", n_threads=n_threads, verbose=verbose)
        success = [1 if r[0] == 0 else 0 for r in results]
        energy_kcal = [r[1] for r in results]
        n_actual_confs = len(results)

        if verbose:
            logger.info(f"{sum(success)}/{n_actual_confs} conformers are optimized.")
            logger.info(f"energies: {energy_kcal}")

        return rkmol, success, energy_kcal
    else:
        return rkmol, None, None


def append_conformers_to_mol(mol: Chem.Mol, conformers: List[np.ndarray]) -> Chem.Mol:
    """append conformers to rkmol. existing conformers in rkmol are unchanged"""
    rkmol = Chem.Mol(mol)
    natoms = rkmol.GetNumAtoms(onlyExplicit=False)
    for coords in conformers:
        assert coords.shape == (natoms, 3)
        rkconf = Chem.Conformer(natoms)
        for atom_idx in range(natoms):
            atom_pos = Point3D(*coords[atom_idx].tolist())
            rkconf.SetAtomPosition(atom_idx, atom_pos)
        rkmol.AddConformer(rkconf, assignId=True)
    return rkmol
