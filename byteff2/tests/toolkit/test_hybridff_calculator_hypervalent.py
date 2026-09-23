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

"""Tests for ``HybridFFCalculator`` on hypervalent molecules (PF6-).

The calculator must xtb-optimize hypervalent molecules in ``__init__`` and
patch the cached PreMMBonded(Conj) parameters so that subsequent ASE-style
single-point evaluations produce non-trivial bonded energies/forces. Without
the patch, ``patch_bonded_terms`` masks every PF6 bond/angle force constant
to 0 and the calculator silently returns wrong energies.
"""

import numpy as np
import pytest

from byteff2.bytemol.core import Molecule


PF6_MAPPED_SMILES = "[F:1][P-:2]([F:3])([F:4])([F:5])([F:6])[F:7]"
ETHANOL_MAPPED_SMILES = "[H:1][O:2][C:3]([H:4])([H:5])[C:6]([H:7])([H:8])[H:9]"


def _load_trained_model():
    from byteff2.train.utils import load_pretrained_model

    return load_pretrained_model("ByteFF-Pol-25")


def _cluster_concat_coords(mols, conformer=0):
    return np.concatenate([mol.conformers[conformer].coords for mol in mols], axis=0)


def test_hybridff_calculator_pf6_patches_bonded_params_and_runs_xtb():
    """For a PF6- input, ``HybridFFCalculator.__init__`` must:

    1. detect the hypervalent center,
    2. run ``XTBCalculator.run_opt`` once to prepare patch coordinates without
       rewriting ``mol.conformers[0]``,
    3. cache PreMMBonded(Conj) ffparams whose entries at hypervalent
       positions are no longer zero (training-path mask=0) but instead
       hold the write-time constants ``bk=1000``, ``ak=278.44`` and the
       r0/d0 inverted from the xtb geometry.
    """
    from byteff2.toolkit.hybridff_calculator import HybridFFCalculator

    model = _load_trained_model()
    model.eval()

    pf6 = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, nconfs=1, name="PF6")
    initial_coords = pf6.conformers[0].coords.copy()

    calc = HybridFFCalculator([pf6], model, cluster=True)

    # 1. xtb-opt was run for the calculator patch coordinates, but the input
    # molecule conformer must not be rewritten.  Otherwise ASE optimization
    # starts from the xTB structure while RMSD is also measured against the
    # mutated structure.
    final_coords = pf6.conformers[0].coords
    np.testing.assert_allclose(final_coords, initial_coords)
    patch_coords = calc.data.coords.squeeze(1).detach().cpu().numpy()
    assert patch_coords.shape == initial_coords.shape
    assert not np.allclose(patch_coords, initial_coords)

    # 2. patch path is enabled and ffparams are cached.
    assert calc._cached_ffparams is not None

    ffp = calc._cached_ffparams
    bk1 = ffp["PreMMBondedConj.bond_k1"].squeeze(-1).detach().cpu().numpy()
    bk2 = ffp["PreMMBondedConj.bond_k2"].squeeze(-1).detach().cpu().numpy()
    ak1 = ffp["PreMMBondedConj.angle_k1"].squeeze(-1).detach().cpu().numpy()
    ak2 = ffp["PreMMBondedConj.angle_k2"].squeeze(-1).detach().cpu().numpy()

    # 3. all PF6 bonds/angles are masked, so the patch constants must hold
    # everywhere in the cached parameter tensors.
    np.testing.assert_allclose(bk1 + bk2, 1000.0, atol=1e-3)
    np.testing.assert_allclose(ak1 + ak2, 278.44, atol=1e-3)
    if "PreMMBonded.bond_k" in ffp:
        bond_k = ffp["PreMMBonded.bond_k"].squeeze(-1).detach().cpu().numpy()
        np.testing.assert_allclose(bond_k, 1000.0, atol=1e-3)
    if "PreMMBonded.angle_k" in ffp:
        angle_k = ffp["PreMMBonded.angle_k"].squeeze(-1).detach().cpu().numpy()
        np.testing.assert_allclose(angle_k, 278.44, atol=1e-3)


def test_hybridff_calculator_cluster_pf6_with_neutral_partner_only_patches_pf6():
    """When PF6 is one of several molecules in the cluster, xtb is called
    only on the hypervalent member; the neutral partner's coordinates and
    its slice of cached PreMMBonded.bond_k must remain non-constant
    (model-predicted, not the patch constant 1000).
    """
    from byteff2.toolkit.hybridff_calculator import HybridFFCalculator

    model = _load_trained_model()
    model.eval()

    pf6 = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, nconfs=1, name="PF6")
    etoh = Molecule.from_mapped_smiles(ETHANOL_MAPPED_SMILES, nconfs=1, name="etoh")

    pf6_initial = pf6.conformers[0].coords.copy()
    etoh_initial = etoh.conformers[0].coords.copy()

    calc = HybridFFCalculator([pf6, etoh], model, cluster=True)

    # Neither input molecule should be mutated by the xTB patch preparation.
    np.testing.assert_allclose(pf6.conformers[0].coords, pf6_initial)
    np.testing.assert_allclose(etoh.conformers[0].coords, etoh_initial)

    # The calculator still uses xTB-optimized PF6 coords for patched bonded
    # parameters, while the neutral partner slice remains the original coords.
    calc_coords = calc.data.coords.squeeze(1).detach().cpu().numpy()
    assert not np.allclose(calc_coords[: pf6.natoms], pf6_initial)
    np.testing.assert_allclose(calc_coords[pf6.natoms :], etoh_initial)

    # patch_bond_mask is concatenated across the cluster; PF6 owns the
    # first 6 bonds (all masked), ethanol owns the rest (all unmasked).
    data = calc.data
    bond_mask = data.patch_bond_mask.detach().cpu().numpy() > 0.9
    assert bond_mask.sum() == 6
    assert (~bond_mask).sum() == data.patch_bond_mask.numel() - 6

    ffp = calc._cached_ffparams
    if "PreMMBonded.bond_k" in ffp:
        bond_k = ffp["PreMMBonded.bond_k"].squeeze(-1).detach().cpu().numpy()
    else:
        bond_k1 = ffp["PreMMBondedConj.bond_k1"].squeeze(-1).detach().cpu().numpy()
        bond_k2 = ffp["PreMMBondedConj.bond_k2"].squeeze(-1).detach().cpu().numpy()
        bond_k = bond_k1 + bond_k2
    # PF6 segment: hard-coded patch constant.
    np.testing.assert_allclose(bond_k[bond_mask], 1000.0, atol=1e-3)
    # Ethanol segment: model-predicted parameters; must not equal the
    # patch constant for every bond.
    assert not np.allclose(bond_k[~bond_mask], 1000.0, atol=1.0)

    # Single-point evaluation on the concatenated coords runs without
    # touching graph_block / preff_block again (covered by cached ffparams).
    coords = _cluster_concat_coords([pf6, etoh])
    energy, forces = calc._calculate_without_restraint(coords)
    assert np.isfinite(energy)
    assert forces.shape == coords.shape
    assert np.all(np.isfinite(forces))


def test_hybridff_calculator_non_hypervalent_uses_cached_params():
    """A purely non-hypervalent input must NOT trigger xtb / patching; the
    calculator should still cache graph/pre-ff params and reuse them on every
    call.
    """
    from byteff2.toolkit.hybridff_calculator import HybridFFCalculator

    model = _load_trained_model()
    model.eval()

    etoh = Molecule.from_mapped_smiles(ETHANOL_MAPPED_SMILES, nconfs=1, name="etoh")
    coords_before = etoh.conformers[0].coords.copy()

    calc = HybridFFCalculator([etoh], model, cluster=True)

    assert calc._cached_ffparams is not None
    assert calc._cached_node_h is not None
    assert calc._cached_edge_h is not None
    # ethanol conformer must not have been touched by xtb.
    np.testing.assert_allclose(etoh.conformers[0].coords, coords_before)

    energy, forces = calc._calculate_without_restraint(coords_before)
    assert np.isfinite(energy)
    assert forces.shape == coords_before.shape
    assert np.all(np.isfinite(forces))


def test_hybridff_calculator_forces_eval_mode():
    torch = pytest.importorskip("torch")
    from byteff2.toolkit.hybridff_calculator import HybridFFCalculator

    class DummyForceField:
        def __init__(self):
            self.eval_called = False
            self.graph_block_called = False
            self.preff_block_called = False

        def eval(self):
            self.eval_called = True
            return self

        def parameters(self):
            yield torch.zeros(1)

        def graph_block(self, _data):
            self.graph_block_called = True
            return "node_h", "edge_h", "xs"

        def preff_block(self, _data, _node_h, _edge_h, ffparams, do_patch=False):
            self.preff_block_called = True
            assert do_patch is False
            return ffparams

    mol = Molecule.from_mapped_smiles(ETHANOL_MAPPED_SMILES, nconfs=1, name="etoh")
    forcefield = DummyForceField()

    calc = HybridFFCalculator([mol], forcefield, cluster=True)

    assert calc.forcefield is forcefield
    assert forcefield.eval_called is True
    assert forcefield.graph_block_called is True
    assert forcefield.preff_block_called is True


def test_hybridff_calculator_hypervalent_without_mm_bonded_skips_xtb(monkeypatch):
    torch = pytest.importorskip("torch")
    from byteff2.toolkit import hybridff_calculator

    def fail_xtb(*_args, **_kwargs):
        raise AssertionError("xTB should not run without an MM bonded layer")

    class DummyForceField:
        def __init__(self):
            self.ff_block = type("DummyFFBlock", (), {"ff_layers": {}})()

        def eval(self):
            return self

        def parameters(self):
            yield torch.zeros(1)

        def graph_block(self, _data):
            return "node_h", "edge_h", "xs"

        def preff_block(self, _data, _node_h, _edge_h, ffparams, do_patch=False):
            assert do_patch is False
            return ffparams

    monkeypatch.setattr(hybridff_calculator, "xtb_optimized_coords", fail_xtb)

    pf6 = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, nconfs=1, name="PF6")
    initial_coords = pf6.conformers[0].coords.copy()

    calc = hybridff_calculator.HybridFFCalculator([pf6], DummyForceField(), cluster=True)

    np.testing.assert_allclose(pf6.conformers[0].coords, initial_coords)
    np.testing.assert_allclose(calc.data.coords.squeeze(1).detach().cpu().numpy(), initial_coords)


def test_hybridff_calculator_records_mol_energy_and_separate_terms():
    torch = pytest.importorskip("torch")
    from byteff2.toolkit.hybridff_calculator import HybridFFCalculator

    class DummyFFBlock:
        ff_layers = {}

        def __call__(self, data, _node_h, _edge_h, ffparams, cluster=False):
            natoms = data.coords.shape[0]
            force_value = 2.0 if cluster else 1.0
            forces = torch.full((natoms, 1, 3), force_value)
            if not cluster:
                return torch.tensor([[3.0]]), forces

            ffparams["DISP"] = torch.tensor([1.0, 2.0])
            ffparams["PAULI"] = torch.tensor([3.0])
            ffparams["ELEC"] = torch.tensor([4.0])
            ffparams["POLARIZATION"] = torch.tensor([5.0])
            ffparams["CHARGE_TRANSFER"] = torch.tensor([0.5])
            ffparams["NNPLayer.energy_cluster"] = torch.tensor([[7.0]])
            ffparams["NNPLayer.energy"] = torch.tensor([[2.0]])
            return torch.tensor([[10.0]]), forces

    class DummyForceField:
        def __init__(self):
            self.ff_block = DummyFFBlock()

        def eval(self):
            return self

        def parameters(self):
            yield torch.zeros(1)

        def graph_block(self, _data):
            return "node_h", "edge_h", "xs"

        def preff_block(self, _data, _node_h, _edge_h, ffparams, do_patch=False):
            assert do_patch is False
            return ffparams

    mol = Molecule.from_mapped_smiles(ETHANOL_MAPPED_SMILES, nconfs=1, name="etoh")
    calc = HybridFFCalculator([mol], DummyForceField(), cluster=True, sep=True)

    energy, forces = calc._calculate_without_restraint(mol.conformers[0].coords)
    separate_energy, separate_forces = calc.get_separate_terms()

    np.testing.assert_allclose(energy, 10.0)
    np.testing.assert_allclose(forces, np.full((mol.natoms, 3), 2.0))
    np.testing.assert_allclose(calc.get_mols_energies(), np.array([3.0]))
    np.testing.assert_allclose(separate_energy["DISP"], 3.0)
    np.testing.assert_allclose(separate_energy["PAULI"], 3.0)
    np.testing.assert_allclose(separate_energy["ELEC"], 4.0)
    np.testing.assert_allclose(separate_energy["POLARIZATION"], 5.0)
    np.testing.assert_allclose(separate_energy["INTER_CHARGE"], 9.0)
    np.testing.assert_allclose(separate_energy["CHARGE_TRANSFER"], 0.5)
    np.testing.assert_allclose(separate_energy["INTER_NNP"], 5.0)
    np.testing.assert_allclose(separate_energy["INTER_VdW"], 11.5)
    assert separate_forces == {}


def test_hybridff_calculator_separate_terms_tolerates_partial_ffparams():
    torch = pytest.importorskip("torch")
    from byteff2.toolkit.hybridff_calculator import HybridFFCalculator

    calc = HybridFFCalculator.__new__(HybridFFCalculator)
    calc.separate_energy = {"DISP": 99.0, "INTER_VdW": 99.0}

    calc._update_separate_terms(
        {
            "NNPLayer.energy_cluster": torch.tensor([[7.0]]),
            "NNPLayer.energy": torch.tensor([[2.0]]),
        }
    )

    assert "DISP" not in calc.separate_energy
    assert "PAULI" not in calc.separate_energy
    assert "ELEC" not in calc.separate_energy
    assert "POLARIZATION" not in calc.separate_energy
    np.testing.assert_allclose(calc.separate_energy["CHARGE_TRANSFER"], 0.0)
    np.testing.assert_allclose(calc.separate_energy["INTER_NNP"], 5.0)
    np.testing.assert_allclose(calc.separate_energy["INTER_VdW"], 5.0)
