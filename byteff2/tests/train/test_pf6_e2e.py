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

"""End-to-end tests for PF6- (hexacoordinate phosphorus) covering both the
training path (do_patch=False, no xtb, gradients masked at hypervalent
positions) and the parameter-writing path (do_patch=True, real xtb gfn2
geometry optimization for bond/angle parameters and itp generation).

Both cases skip gracefully when the trained checkpoint or the xtb binary is
unavailable.
"""

import json
import math
import os

import numpy as np
import pytest
import torch

from byteff2.bytemol.core import Molecule


PF6_MAPPED_SMILES = "[F:1][P-:2]([F:3])([F:4])([F:5])([F:6])[F:7]"
ETHANOL_MAPPED_SMILES = "[H:1][O:2][C:3]([H:4])([H:5])[C:6]([H:7])([H:8])[H:9]"
HYPERS_SMILES = "[S]1(F)(F)(F)(F)OCCO1"
HYPERS_DIMETHOXY_SMILES = "CO[S](F)(F)(F)(F)OC"
_BONDED_REFERENCE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "testdata",
    "hypervalent_get_nb_params_bonded_reference.json",
)


def _load_trained_model():
    from byteff2.train.utils import load_pretrained_model

    try:
        return load_pretrained_model("ByteFF-Pol-25")
    except FileNotFoundError as error:
        pytest.skip(str(error))


def _premmbondedconj_weight_grad_sum(model) -> float:
    """Sum of |grad| over all PreMMBondedConj parameters."""
    total = 0.0
    for n, p in model.named_parameters():
        if "PreMMBondedConj" in n and p.grad is not None:
            total += p.grad.abs().sum().item()
    return total


def _hypervalent_atom_indices(mol, atomic_nums=(15, 16), min_degree=5):
    return {
        atom.GetIdx()
        for atom in mol.rkmol.GetAtoms()
        if atom.GetDegree() >= min_degree and atom.GetAtomicNum() in atomic_nums
    }


def _backward_bonded_conj_loss(model, mol):
    """Run the training-path forward (do_patch=False) and back-propagate a
    synthetic loss that depends only on the PreMMBondedConj outputs, so the
    test isolates whether hypervalent positions get masked out."""
    from byteff2.data import GraphData

    confdata = {"coords": torch.tensor(mol.conformers[0].coords, dtype=torch.float32).unsqueeze(0)}
    data = GraphData(mol.name, mol.get_mapped_smiles(), record_nonbonded_all=False, confdata=confdata, max_n_confs=1)

    preds = model(data, skip_ff=True, do_patch=False, validate_elements=False)
    ff = preds["ff_parameters"]
    loss = sum(
        ff[k].pow(2).sum()
        for k in (
            "PreMMBondedConj.bond_k1",
            "PreMMBondedConj.bond_k2",
            "PreMMBondedConj.angle_k1",
            "PreMMBondedConj.angle_k2",
        )
    )
    loss.backward()
    return data


def _conj_to_bond_r0(k1, k2):
    """Recover bond r0 (Angstrom) from PreMMBondedConj k1/k2.

    PreMMBondedConj uses a conjugate basis at b1=0.5, b2=4.0 such that for
    a target equilibrium length r0:
        k2 = (r0 - b1) / (b2 - b1) * (k1 + k2),
    therefore r0 = b1 + (b2 - b1) * k2 / (k1 + k2).
    Reference: byteff2/model/ff_layers/mm_bonded.py PreMMBondedConj.bond_b1/b2.
    """
    b1, b2 = 0.5, 4.0
    return b1 + (b2 - b1) * (k2 / (k1 + k2))


def _conj_to_angle_d0_deg(k1, k2):
    """Recover angle d0 (degrees) from PreMMBondedConj k1/k2.

    The conjugate basis is b1=0.25*pi, b2=1.05*pi (radians).
    """
    b1, b2 = 0.25 * math.pi, 1.05 * math.pi
    radians = b1 + (b2 - b1) * (k2 / (k1 + k2))
    return np.degrees(radians)


def _bond_lengths(coords, idx):
    coords = np.asarray(coords)
    idx = np.asarray(idx)
    diffs = coords[idx[:, 0]] - coords[idx[:, 1]]
    return np.linalg.norm(diffs, axis=1)


def _angles_deg(coords, idx):
    coords = np.asarray(coords)
    idx = np.asarray(idx)
    a = coords[idx[:, 0]] - coords[idx[:, 1]]
    c = coords[idx[:, 2]] - coords[idx[:, 1]]
    cos = (a * c).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _sort_bonded_param_records(records):
    return sorted(records, key=lambda record: tuple(record.items()))


def _itp_bonded_params(tfs):
    from byteff2.bytemol.toolkit.gmxtool.topparse import DihedralTypeEnum

    mol_topo = tfs.mol_topos[0]
    bonds = [{"ai": r.ai, "aj": r.aj, "funct": r.funct.name, "c0": r.c0, "c1": r.c1} for r in mol_topo.bonds]
    angles = [
        {"ai": r.ai, "aj": r.aj, "ak": r.ak, "funct": r.funct.name, "c0": r.c0, "c1": r.c1} for r in mol_topo.angles
    ]
    propers = []
    impropers = []
    for r in mol_topo.dihedrals:
        params = {
            "ai": r.ai,
            "aj": r.aj,
            "ak": r.ak,
            "al": r.al,
            "funct": r.funct.name,
            "c0": r.c0,
            "c1": r.c1,
            "c2": r.c2,
        }
        if r.funct == DihedralTypeEnum.MULTIPLE_PROPER:
            propers.append(params)
        elif r.funct == DihedralTypeEnum.PERIODIC_IMPROPER:
            impropers.append(params)
        else:
            raise AssertionError(f"unexpected dihedral function {r.funct}")
    return {
        "bonds": _sort_bonded_param_records(bonds),
        "angles": _sort_bonded_param_records(angles),
        "propers": _sort_bonded_param_records(propers),
        "impropers": _sort_bonded_param_records(impropers),
    }


def _assert_itp_bonded_params_match_reference(actual, expected):
    assert actual.keys() == expected.keys()
    for section in actual:
        assert len(actual[section]) == len(expected[section]), section
        for actual_record, expected_record in zip(actual[section], expected[section], strict=True):
            assert actual_record.keys() == expected_record.keys()
            for key in actual_record:
                if isinstance(expected_record[key], float):
                    np.testing.assert_allclose(actual_record[key], expected_record[key], rtol=1e-6, atol=1e-2)
                else:
                    assert actual_record[key] == expected_record[key]


def test_pf6_training_path_zero_gradient_at_hypervalent():
    """Training path: do_patch=False; with the synthetic loss above, the
    PreMMBondedConj parameter gradients on the fully-hypervalent PF6- must
    be exactly zero (torch.where short-circuits the gradient), while a
    normal molecule (ethanol) must still produce nonzero gradient.
    """
    torch.manual_seed(0)

    # PF6- (every bond/angle is masked).
    model = _load_trained_model()
    model.train()
    pf6 = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, nconfs=1, name="PF6")
    data_pf6 = _backward_bonded_conj_loss(model, pf6)
    assert int(data_pf6.patch_bond_mask.sum().item()) == data_pf6.patch_bond_mask.numel() == 6
    assert int(data_pf6.patch_angle_mask.sum().item()) == data_pf6.patch_angle_mask.numel() == 15
    grad_pf6 = _premmbondedconj_weight_grad_sum(model)
    assert grad_pf6 == 0.0, f"PF6 PreMMBondedConj weights should have zero grad, got {grad_pf6}"

    # Ethanol control: no hypervalent atoms; the same synthetic loss must
    # produce nonzero gradient through the unmasked bond/angle outputs.
    model = _load_trained_model()
    model.train()
    etoh = Molecule.from_mapped_smiles(ETHANOL_MAPPED_SMILES, nconfs=1, name="etoh")
    data_etoh = _backward_bonded_conj_loss(model, etoh)
    assert int(data_etoh.patch_bond_mask.sum().item()) == 0
    grad_etoh = _premmbondedconj_weight_grad_sum(model)
    assert grad_etoh > 0.0, f"ethanol PreMMBondedConj weights should have nonzero grad, got {grad_etoh}"


@pytest.mark.parametrize(
    ("smiles", "name"),
    [
        (HYPERS_SMILES, "hyperS"),
        (HYPERS_DIMETHOXY_SMILES, "hyperS_dimethoxy"),
    ],
)
def test_hypervalent_sulfur_training_path_partial_mask_keeps_normal_gradient(smiles, name):
    """Training path on a partially-hypervalent molecule (S(F)4(O)2 ring).

    Only bonds/angles incident to the central S are masked; the C-O / C-C /
    C-H part still contributes gradient. This ensures torch.where masks
    only the targeted positions and not the whole molecule.
    """
    torch.manual_seed(0)

    model = _load_trained_model()
    model.train()
    mol = Molecule.from_smiles(smiles, nconfs=1, name=name)
    data = _backward_bonded_conj_loss(model, mol)

    # masks should be partially set (some bonds/angles touch the
    # hypervalent S, others do not).
    special = _hypervalent_atom_indices(mol)
    assert special, "expected at least one hypervalent atom in the test SMILES"

    expected_bond_mask = torch.tensor(
        [float(bond[0].item() in special or bond[1].item() in special) for bond in data.inc_node_bond],
        dtype=torch.float32,
    )
    expected_angle_mask = torch.tensor(
        [float(angle[1].item() in special) for angle in data.inc_node_angle],
        dtype=torch.float32,
    )
    assert torch.equal(data.patch_bond_mask, expected_bond_mask)
    assert torch.equal(data.patch_angle_mask, expected_angle_mask)
    assert 0 < int(expected_bond_mask.sum().item()) < expected_bond_mask.numel()
    assert 0 < int(expected_angle_mask.sum().item()) < expected_angle_mask.numel()

    # because non-S parts are not masked, PreMMBondedConj weights should
    # still receive nonzero gradient overall.
    grad = _premmbondedconj_weight_grad_sum(model)
    assert grad > 0.0, f"partially-masked molecule should still produce gradient, got {grad}"


def test_pf6_write_params_runs_xtb_and_emits_itp(tmp_path):
    """Parameter-writing path: do_patch=True triggers XTBCalculator.run_opt
    (xtb gfn2 geometry optimization), then write_params produces an itp +
    json set for PF6-.

    This test is heavy by design: it exercises the full chain end-to-end.
    """
    from byteff2.train.utils import _get_hypervalent_patched_ffparams, get_nb_params

    model = _load_trained_model()
    model.eval()

    mol = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, name="PF6")
    assert not mol.conformers

    # Capture the same write-time patched PreMMBondedConj parameters through
    # the helper used by get_nb_params, without changing get_nb_params' public
    # return value.
    patch_mol = Molecule.from_mapped_smiles(PF6_MAPPED_SMILES, nconfs=1, name="PF6")
    data, ff = _get_hypervalent_patched_ffparams(model, patch_mol)
    bk1 = ff["PreMMBondedConj.bond_k1"].squeeze(-1).detach().cpu().numpy()
    bk2 = ff["PreMMBondedConj.bond_k2"].squeeze(-1).detach().cpu().numpy()
    ak1 = ff["PreMMBondedConj.angle_k1"].squeeze(-1).detach().cpu().numpy()
    ak2 = ff["PreMMBondedConj.angle_k2"].squeeze(-1).detach().cpu().numpy()
    proper_k = ff["PreMMBondedConj.proper_k"].detach().cpu().numpy()
    bond_idx = data.inc_node_bond.cpu().numpy()
    angle_idx = data.inc_node_angle.cpu().numpy()
    opt_coords = data.coords.squeeze(1).cpu().numpy()

    # 1. The write-time patch hardcodes the total bonded force constants. For
    # PF6- (every bond/angle masked): k1 + k2 must equal bk = 1000
    # (kcal/mol/A^2) and ak = 278.44 (kcal/mol/rad^2).
    np.testing.assert_allclose(bk1 + bk2, 1000.0, atol=1e-3)
    np.testing.assert_allclose(ak1 + ak2, 278.44, atol=1e-3)

    # 2. Inverting the conjugate basis recovers r0 / d0 that match the
    # xtb-optimized geometry (PF6- target: octahedral, ~1.6 A bonds,
    # 90 / 180 deg angles).
    r0 = _conj_to_bond_r0(bk1, bk2)
    d0 = _conj_to_angle_d0_deg(ak1, ak2)
    bond_lens = _bond_lengths(opt_coords, bond_idx)
    angle_vals = _angles_deg(opt_coords, angle_idx)
    np.testing.assert_allclose(r0, bond_lens, atol=2e-2)
    np.testing.assert_allclose(d0, angle_vals, atol=1.0)
    # All P-F bond lengths should cluster around the expected octahedral value.
    assert 1.50 < bond_lens.min() <= bond_lens.max() < 1.75
    # 12 angles ~ 90 deg (cis) and 3 angles ~ 180 deg (trans).
    assert int((np.abs(angle_vals - 90.0) < 5.0).sum()) == 12
    assert int((np.abs(angle_vals - 180.0) < 5.0).sum()) == 3

    # 3. PF6- is symmetric octahedral and has no proper torsions defined,
    # so all proper_k rows that touch the hypervalent P should be zero.
    proper_quads = data.inc_node_proper.cpu().numpy()
    p_idx = next(i for i, z in enumerate(patch_mol.atomic_numbers) if z == 15)
    hyper_rows = np.array([p_idx in q for q in proper_quads], dtype=bool)
    if hyper_rows.any():
        np.testing.assert_allclose(proper_k[hyper_rows], 0.0, atol=1e-9)

    _, params, tfs, mol = get_nb_params(model, mol, write_to_itp=True)
    final_coords = torch.tensor(mol.conformers[0].coords, dtype=torch.float32)

    # 4. Charges sum to the formal charge (-1) for PF6-.
    assert pytest.approx(sum(params["charge"]), abs=1e-3) == -1.0
    # The phosphorus center carries a positive partial charge while the
    # six fluorines are negative.
    assert params["charge"][1] > 0.0  # P
    for f_charge in [params["charge"][i] for i in (0, 2, 3, 4, 5, 6)]:
        assert f_charge < 0.0

    # 5. get_nb_params generates a conformer (none was attached), runs
    # XTBCalculator.run_opt, and writes the optimized geometry into
    # mol.conformers[0].coords.
    assert mol.conformers
    assert final_coords.shape == (7, 3)

    # 6. Write itp/gro/json into a tmp dir and check the files are present
    # and non-trivial.
    out_dir = tmp_path / "PF6_out"
    out_dir.mkdir()
    itp_path = out_dir / "PF6.itp"
    json_path = out_dir / "PF6.json"
    tfs.write_itp(str(itp_path), separated_atp=True)
    with open(json_path, "w") as f:
        json.dump(params, f)

    assert itp_path.is_file() and itp_path.stat().st_size > 0
    itp_text = itp_path.read_text()
    assert "[ atoms ]" in itp_text
    assert "[ bonds ]" in itp_text
    assert "[ angles ]" in itp_text
    # PF6-: 6 P-F bonds, 15 F-P-F angles. Use a soft check on counts via the
    # number of non-comment lines in each section.
    bond_lines = [
        ln
        for ln in itp_text.split("[ bonds ]", 1)[1].split("[ angles ]", 1)[0].splitlines()
        if ln and not ln.startswith(";")
    ]
    angle_lines = [
        ln for ln in itp_text.split("[ angles ]", 1)[1].splitlines() if ln and not ln.startswith(";") and ln.strip()
    ]
    assert len(bond_lines) == 6
    assert len(angle_lines) == 15


def test_hypervalent_sulfur_write_params_runs_xtb_and_emits_itp(tmp_path):
    """Parameter-writing path on the hypervalent S ring: do_patch=True
    triggers XTBCalculator.run_opt for a partially hypervalent molecule
    and write_params produces an itp/json set with bond/angle parameters
    derived from the GFN2-optimized geometry on the hypervalent positions
    while proper terms touching the hypervalent center are zeroed.
    """
    from byteff2.train.utils import _get_hypervalent_patched_ffparams, get_nb_params

    model = _load_trained_model()
    model.eval()

    mol = Molecule.from_smiles(HYPERS_SMILES, nconfs=1, name="hyperS")
    initial_coords = torch.tensor(mol.conformers[0].coords, dtype=torch.float32).clone()
    natoms = mol.natoms

    patch_mol = Molecule.from_smiles(HYPERS_SMILES, nconfs=1, name="hyperS")
    data, ff = _get_hypervalent_patched_ffparams(model, patch_mol)
    bk1 = ff["PreMMBondedConj.bond_k1"].squeeze(-1).detach().cpu().numpy()
    bk2 = ff["PreMMBondedConj.bond_k2"].squeeze(-1).detach().cpu().numpy()
    ak1 = ff["PreMMBondedConj.angle_k1"].squeeze(-1).detach().cpu().numpy()
    ak2 = ff["PreMMBondedConj.angle_k2"].squeeze(-1).detach().cpu().numpy()
    proper_k = ff["PreMMBondedConj.proper_k"].detach().cpu().numpy()
    improper_k = ff["PreMMBondedConj.improper_k"].detach().cpu().numpy()
    bond_mask = data.patch_bond_mask.cpu().numpy() > 0.9
    angle_mask = data.patch_angle_mask.cpu().numpy() > 0.9
    proper_mask = data.patch_proper_mask.cpu().numpy() > 0.9
    improper_mask = data.patch_improper_mask.cpu().numpy() > 0.9
    assert bond_mask.any() and (~bond_mask).any(), "hyperS should have both masked and unmasked bonds"
    assert angle_mask.any() and (~angle_mask).any(), "hyperS should have both masked and unmasked angles"
    assert proper_mask.any(), "hyperS ring propers touching S should be masked"

    bond_idx = data.inc_node_bond.cpu().numpy()
    angle_idx = data.inc_node_angle.cpu().numpy()
    opt_coords = data.coords.squeeze(1).cpu().numpy()

    # Hypervalent positions: k1+k2 must be the hard-coded constants and
    # the recovered r0/d0 must match xtb-optimized geometry.
    np.testing.assert_allclose(bk1[bond_mask] + bk2[bond_mask], 1000.0, atol=1e-3)
    np.testing.assert_allclose(ak1[angle_mask] + ak2[angle_mask], 278.44, atol=1e-3)

    bond_lens = _bond_lengths(opt_coords, bond_idx)
    angle_vals = _angles_deg(opt_coords, angle_idx)
    r0_hyper = _conj_to_bond_r0(bk1[bond_mask], bk2[bond_mask])
    d0_hyper = _conj_to_angle_d0_deg(ak1[angle_mask], ak2[angle_mask])
    np.testing.assert_allclose(r0_hyper, bond_lens[bond_mask], atol=3e-2)
    np.testing.assert_allclose(d0_hyper, angle_vals[angle_mask], atol=2.0)
    np.testing.assert_allclose(proper_k[proper_mask], 0.0, atol=1e-9)
    if improper_mask.any():
        np.testing.assert_allclose(improper_k[improper_mask], 0.0, atol=1e-9)

    # Non-hypervalent positions: must NOT be hard-coded; k1+k2 generally
    # disagrees with the patched constants.
    if (~bond_mask).any():
        sums = bk1[~bond_mask] + bk2[~bond_mask]
        assert not np.allclose(sums, 1000.0, atol=1.0), (
            "non-hypervalent bonds should not be replaced by the patch constant"
        )

    _, params, tfs, mol = get_nb_params(model, mol, write_to_itp=True)
    final_coords = torch.tensor(mol.conformers[0].coords, dtype=torch.float32)

    assert final_coords.shape == (natoms, 3)
    assert not torch.allclose(final_coords, initial_coords)

    out_dir = tmp_path / "hyperS_out"
    out_dir.mkdir()
    itp_path = out_dir / "hyperS.itp"
    json_path = out_dir / "hyperS.json"
    tfs.write_itp(str(itp_path), separated_atp=True)
    with open(json_path, "w") as f:
        json.dump(params, f)

    assert itp_path.is_file() and itp_path.stat().st_size > 0
    itp_text = itp_path.read_text()
    assert "[ atoms ]" in itp_text
    assert "[ bonds ]" in itp_text
    assert "[ angles ]" in itp_text
    assert "[ dihedrals ]" in itp_text


@pytest.mark.parametrize(
    ("smiles", "name", "mapped", "should_raise"),
    [
        (PF6_MAPPED_SMILES, "PF6", True, False),
        (HYPERS_SMILES, "hyperS", False, False),
        (HYPERS_DIMETHOXY_SMILES, "hyperS_dimethoxy", False, True),
    ],
)
def test_get_nb_params_hypervalent_write_path(smiles, name, mapped, should_raise):
    """get_nb_params should emit stable ITP bonded params for supported
    hypervalent molecules and reject non-ring hypervalent S centers explicitly.
    """
    from byteff2.train.utils import get_nb_params

    model = _load_trained_model()
    model.eval()

    mol_factory = Molecule.from_mapped_smiles if mapped else Molecule.from_smiles
    actual_mol = mol_factory(smiles, nconfs=1, name=name)

    if should_raise:
        with pytest.raises(ValueError, match="non-ring hypervalent atoms cannot be patched"):
            get_nb_params(model, actual_mol, write_to_itp=True)
        return

    with open(_BONDED_REFERENCE_JSON) as f:
        references = json.load(f)
    _, _, tfs, _ = get_nb_params(model, actual_mol, write_to_itp=True)
    _assert_itp_bonded_params_match_reference(_itp_bonded_params(tfs), references[name])
