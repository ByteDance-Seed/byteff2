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

import os

import numpy as np
import torch

from byteff2.data import ClusterData, collate_data, MonoData
from byteff2.model.ff_layers.ff_kernels import ClassicalForceField as CFF
from byteff2.model.ff_layers.ff_opt import ConstraintFFopt
from byteff2.model.ff_layers.utils import dihedral_jacobian
from byteff2.utils.definitions import MMParam, MMTerm


def test_adjust_position_max_iter_zero():
    """Regression: max_iter=0 must not raise UnboundLocalError on
    `converge_flag` / `cflag` and should fall back to no position update."""

    natoms, nconfs, bs = 3, 1, 1
    coords = torch.zeros((natoms, nconfs, 3))
    update = torch.zeros_like(coords)
    counts = torch.tensor([natoms])

    def jac_fn(c: torch.Tensor):
        return torch.zeros((bs, nconfs)), torch.zeros_like(c)

    delta, converge_flag = ConstraintFFopt.adjust_position(
        coords,
        update,
        counts,
        jac_fn,
        max_iter=0,
        threshold=1e-5,
    )

    # No crash, shapes are sane, and the fallback marks all rows as not-converged.
    assert delta.shape == coords.shape
    assert converge_flag.shape == (bs, nconfs)
    assert converge_flag.dtype == torch.bool
    assert not converge_flag.any()
    # With max_iter=0 we keep tmp_coords == coords, so delta is all zeros.
    assert torch.equal(delta, torch.zeros_like(delta))


def _build_cluster_data(nmols: int = 3, max_n_confs: int = 2) -> ClusterData:
    """A small ClusterData of `nmols` methane molecules. Cluster counts must
    differ from per-molecule counts so that mis-routed `cluster` flags would
    surface as shape mismatches."""

    smiles = "[C:1]([H:2])([H:3])([H:4])[H:5]"
    natoms_per_mol = 5
    n_node = natoms_per_mol * nmols
    n_conf = max_n_confs
    rng = np.random.default_rng(0)
    # Spread molecules along x so inter-molecular distances are non-trivial.
    base = np.zeros((n_conf, n_node, 3), dtype=np.float32)
    for i in range(nmols):
        for a in range(natoms_per_mol):
            base[:, i * natoms_per_mol + a] = [i * 3.0 + 0.3 * a, 0.0, 0.0]
    base += rng.standard_normal(base.shape).astype(np.float32) * 0.1
    return ClusterData(
        "test",
        mapped_smiles=[smiles] * nmols,
        confdata={"coords": base},
        max_n_confs=max_n_confs,
    )


def test_ljes_forward_cluster_uses_cluster_topology():
    """Regression for mm_nonbonded.LJEs.forward propagating `cluster=True`
    to nonbonded14/nonbonded_all topology and counts. With `cluster=False`
    on a multi-molecule ClusterData the cluster nonbonded_all pairs would be
    skipped (and shapes would be wrong), so the two paths must differ."""

    from byteff2.model.ff_layers import LJEs, PreLJEs

    torch.manual_seed(0)

    data = _build_cluster_data(nmols=3, max_n_confs=2)

    # Sanity: cluster topology really differs from per-molecule one.
    assert (
        data.get_count("nonbonded_all", idx=None, cluster=True).item()
        > data.get_count("nonbonded_all", idx=None).sum().item()
    )

    dim = 16
    n_node = data.node_features.shape[0]
    n_edge = data.edge_features.shape[0]
    x_h = torch.rand((n_node, dim))
    e_h = torch.rand((n_edge, dim))

    pre = PreLJEs(dim, dim)
    params = pre(data, x_h, e_h)

    layer = LJEs(dim, dim)

    # cluster=True path must run end-to-end with shapes aligned.
    energy_c, forces_c = layer(data, x_h, e_h, params, cluster=True)
    n_atom_cluster = int(data.get_count("node", idx=None, cluster=True).item())
    nconfs = data.coords.shape[1]
    assert energy_c.shape == data.confmask_cluster.shape
    assert forces_c.shape == (n_atom_cluster, nconfs, 3)

    # cluster=False on the same ClusterData uses per-molecule counts/topology;
    # the resulting energy must differ from the cluster path because the
    # inter-molecular pairs are excluded.
    energy_m, _ = layer(data, x_h, e_h, params, cluster=False)
    assert not torch.allclose(energy_c, energy_m)


def test_classical_force_field_energy_force_cluster_natoms():
    """Regression for ff_kernels.energy_force using cluster-aware natoms when
    multiplying confmask into forces. With the bug, batch_to_atoms(confmask,
    natoms) would broadcast against the per-molecule node count and crash on
    `forces * confmask_forces`."""

    torch.manual_seed(0)

    data = _build_cluster_data(nmols=3, max_n_confs=2)

    # Build minimal ff_params keyed by MMParam.* (CFF.energy_force expects this
    # flat layout). All-zero parameters are sufficient: the regression target
    # is shape alignment between forces and confmask_forces, not numerics.
    n_bond = data.inc_node_bond.shape[0]
    ff_params = {
        MMParam.bond_k: torch.zeros((n_bond, 1)),
        MMParam.bond_r0: torch.zeros((n_bond, 1)),
    }

    out = CFF.energy_force(
        data,
        ff_params,
        calc_terms=[MMTerm.bond],
        cluster=True,
        confmask=data.confmask_cluster,
    )

    n_atom_cluster = int(data.get_count("node", idx=None, cluster=True).item())
    nconfs = data.coords.shape[1]
    assert out["total_forces"].shape == (n_atom_cluster, nconfs, 3)
    assert out["total_energy"].shape == data.confmask_cluster.shape


# ---------------------------------------------------------------------------
# Batched FFopt vs per-molecule FFopt consistency
# ---------------------------------------------------------------------------
# Following the example workflow in `example/ByteFF/1_training/`, the
# trainer feeds a batch of MonoData (collated via `collate_data`) into
# `ConstraintFFopt.optimize` with the *full* MM force field (all bonded +
# nonbonded14 + nonbonded_all) and a real torsion-angle constraint.
# Internally `BatchedLBFGS` keeps an independent history per row, so
# optimizing N molecules together must give the same coordinates as
# optimizing each molecule on its own.

# Source data: `torsion_example.csv` / `torsion_example.h5` are a 5-molecule
# subset of the ByteFF HuggingFace example dataset, kept under
# `byteff2/tests/testdata/torsion_example/` so the test is hermetic and
# exercises every MM term.
_TORSION_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "testdata", "torsion_example")
_TORSION_CSV = os.path.join(_TORSION_DATA_DIR, "torsion_example.csv")
_TORSION_H5 = os.path.join(_TORSION_DATA_DIR, "torsion_example.h5")
_NUM_TEST_MOLS = 3


# Trained models are resolved from the external asset root by load_pretrained_model.
def _load_torsion_examples(num_mols: int):
    """Return a list of `(mapped_smiles, coords[natoms,3], torsion_ids[4])`
    for the smallest `num_mols` molecules in `torsion_example.h5`."""
    import csv

    import h5py

    with open(_TORSION_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    examples = []
    with h5py.File(_TORSION_H5, "r") as h5:
        cands = []
        for row in rows:
            uuid = row["uuid"]
            if uuid not in h5:
                continue
            grp = h5[uuid]
            natoms = int(grp["atomic_numbers"].shape[0])
            cands.append((natoms, uuid, row["mapped_isomeric_smiles"]))
        cands.sort(key=lambda x: x[0])

        for _, uuid, smi in cands[:num_mols]:
            grp = h5[uuid]
            torsion_ids = np.asarray(grp["torsion_atom_indices"][()], dtype=np.int64)
            # Use the first available constraint geometry as the initial conformer.
            coords = np.asarray(grp["constraint 0"]["coords"][()], dtype=np.float64)
            examples.append((smi, coords, torsion_ids))
    return examples


def _load_trained_gnn_model():
    """Lazy-load the trained model used by `Trainer.ffopt`. Skip the test
    if the checkpoint isn't available locally."""
    import pytest

    from byteff2.train.utils import load_pretrained_model

    try:
        model = load_pretrained_model("ByteFF-Pol-25")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    model.eval()
    return model


def _make_mono_data_from_coords(name: str, mapped_smiles: str, coords: np.ndarray, torsion_ids: np.ndarray) -> MonoData:
    """Build a single-conformer MonoData and attach torsion_ids so that
    `inc_node_torsion_ids` is populated (needed by `dihedral_jacobian`)."""
    confdata = {"coords": coords[None, :, :].astype(np.float32)}  # [1, natoms, 3]
    graph = MonoData(
        name=name,
        mapped_smiles=mapped_smiles,
        confdata=confdata,
        max_n_confs=1,
    )
    # Mirror the layout produced by IMDataset for torsion tasks: a single row
    # of [a0, a1, a2, a3] which is later turned into `inc_node_torsion_ids`
    # via the same hook used in MonoData.__init__.
    graph.torsion_ids = torch.tensor(torsion_ids, dtype=graph.counts.dtype).reshape(1, 4)
    graph.inc_node_torsion_ids = graph.torsion_ids.clone().detach().to(graph.counts.dtype)
    graph.set_count("torsion_ids", graph.torsion_ids.shape[0])
    return graph


def _run_full_ffopt(model, graph) -> torch.Tensor:
    """Run `ConstraintFFopt.optimize` on a graph (MonoData or collated batch)
    using exactly the same code path as `FFJointTrainer.ffopt`: full
    end-to-end model forward (`skip_ff=False`) for energies/forces, real
    dihedral constraint, non-zero position restraint."""
    torsion_ids = graph.inc_node_torsion_ids

    def energy_func(_coords):
        graph.coords = _coords
        with torch.no_grad():
            ff_results = model(graph, skip_ff=False, validate_elements=False)
        return ff_results["energy"], ff_results["forces"]

    def jacobian_func(_coords):
        return dihedral_jacobian(_coords, torsion_ids)

    opt_coords, _, _ = ConstraintFFopt.optimize(
        graph,
        energy_func=energy_func,
        jacobian_func=jacobian_func,
        pos_res_k=1.0,
        max_iter=500,
        max_step=0.05,
        history_size=20,
        f_max=0.05,
    )
    return opt_coords


def test_ffopt_batched_matches_per_molecule():
    """End-to-end: run the trainer-style FFopt (GNN-predicted params + full
    MM + dihedral constraint + position restraint) on 5 molecules from
    `torsion_example.h5`, both independently and as a single collated
    batch.  Per-molecule slices of the batched output must match the
    independent results."""

    torch.manual_seed(0)

    model = _load_trained_gnn_model()
    examples = _load_torsion_examples(_NUM_TEST_MOLS)
    assert len(examples) == _NUM_TEST_MOLS

    # Per-molecule path.
    mono_graphs = [
        _make_mono_data_from_coords(f"mol_{i}", smi, coords, tids) for i, (smi, coords, tids) in enumerate(examples)
    ]
    per_mol_coords = [_run_full_ffopt(model, g) for g in mono_graphs]

    # Batched path: rebuild the same graphs, collate, optimize once.
    mono_graphs_batched = [
        _make_mono_data_from_coords(f"mol_{i}", smi, coords, tids) for i, (smi, coords, tids) in enumerate(examples)
    ]
    batch_graph = collate_data(mono_graphs_batched)
    batched_coords = _run_full_ffopt(model, batch_graph)

    # Slice per-molecule and compare.  Each row's BFGS may stop at a slightly
    # different iteration in the batched run vs the single-molecule run (the
    # batched run can keep iterating until *every* row converges), so the
    # comparison tolerates small fp32 drift around the optimum.
    natoms_per_mol = batch_graph.get_count("node", idx=None).tolist()
    offset = 0
    for i, na in enumerate(natoms_per_mol):
        sliced = batched_coords[offset : offset + na]
        torch.testing.assert_close(sliced, per_mol_coords[i], atol=5e-3, rtol=5e-3)
        offset += na
