from types import SimpleNamespace

import torch

from byteff2.bytemol.core import Molecule
from byteff2.model.ff_layers.mm_bonded import PROPERTORSION_TERMS
from byteff2.toolkit import gmxtool
from byteff2.toolkit.gmxtool import ffparams_to_tfs, patch_hbond14_angle_force_constants


BAD_CASE_MAPPED_SMILES = (
    "[C:1]([O:2][P@@:3](=[O:4])([O-:5])[N:6]([c:7]1[c:9]([H:11])[c:12]([H:13])"
    "[c:14]([C:15]([H:22])([H:23])[H:24])[c:16]([H:17])[c:10]1[H:18])[H:8])"
    "([H:19])([H:20])[H:21]"
)


def _angle(ai, aj, ak, force_constant):
    return SimpleNamespace(ai=ai, aj=aj, ak=ak, k=force_constant)


def _empty_bonded_data():
    """A GraphData stand-in with no bonded terms, so ffparams_to_tfs only emits atoms."""
    return SimpleNamespace(
        inc_node_bond=torch.zeros((0, 2), dtype=torch.long),
        inc_node_angle=torch.zeros((0, 3), dtype=torch.long),
        inc_node_proper=torch.zeros((0, 4), dtype=torch.long),
        inc_node_improper=torch.zeros((0, 4), dtype=torch.long),
        inc_node_nonbonded14=torch.zeros((0, 2), dtype=torch.long),
    )


def _ffparams(mol):
    natoms = mol.natoms
    return {
        "PreLJEs.sigma": torch.zeros(natoms),
        "PreLJEs.epsilon": torch.zeros(natoms),
        "PreLJEs.charges": torch.zeros(natoms),
        "PreMMBonded.bond_k": torch.zeros(0),
        "PreMMBonded.bond_r0": torch.zeros(0),
        "PreMMBonded.angle_k": torch.zeros(0),
        "PreMMBonded.angle_d0": torch.zeros(0),
        "PreMMBonded.proper_k": torch.zeros((0, PROPERTORSION_TERMS)),
        "PreMMBonded.improper_k": torch.zeros(0),
    }


def _stub_tfs_build(monkeypatch, fake_tfs):
    """Short-circuit the record -> TopoFullSystem conversion so tests keep our fake tfs."""
    monkeypatch.setattr(gmxtool.TopoFullSystem, "from_records", lambda **_kwargs: fake_tfs)
    monkeypatch.setattr(
        gmxtool,
        "TopoDefaults",
        lambda _uuid: SimpleNamespace(
            nbfunc=gmxtool.NonbondedFunctionEnum.LENNARD_JONES,
            comb_rule=gmxtool.LJCombinationRuleEnum.SIGMA_EPSILON,
            gen_pairs="yes",
            fudge_lj=0.5,
            fudge_qq=0.5,
        ),
    )


def test_patch_hbond14_angle_force_constants_only_changes_matching_low_angles():
    mol = Molecule.from_mapped_smiles(BAD_CASE_MAPPED_SMILES, nconfs=0)
    angles = [
        _angle(6, 3, 2, 491.0),  # matching N(H)-P-O angle, with reversed endpoints
        _angle(4, 3, 6, 727.0),  # matching angle already above the floor
        _angle(5, 3, 6, 600.0),  # matching angle exactly at the floor
        _angle(2, 3, 4, 491.0),  # O-P-O does not match the SMARTS
    ]
    tfs = SimpleNamespace(mol_topos=[SimpleNamespace(angles=angles)])

    patched_tfs = patch_hbond14_angle_force_constants(tfs, mol)

    assert patched_tfs is tfs
    assert [angle.k for angle in angles] == [600.0, 727.0, 600.0, 491.0]


def test_ffparams_to_tfs_applies_hbond14_patch_by_default(monkeypatch):
    mol = Molecule.from_mapped_smiles(BAD_CASE_MAPPED_SMILES, nconfs=0)
    angle = _angle(6, 3, 2, 491.0)  # matching N(H)-P-O angle below the floor
    fake_tfs = SimpleNamespace(uuid="x", mol_topos=[SimpleNamespace(angles=[angle])])
    _stub_tfs_build(monkeypatch, fake_tfs)

    result = ffparams_to_tfs(_ffparams(mol), _empty_bonded_data(), mol, mol_name="TEST")

    assert result is fake_tfs
    assert angle.k == 600.0


def test_ffparams_to_tfs_skips_hbond14_patch_when_disabled(monkeypatch):
    mol = Molecule.from_mapped_smiles(BAD_CASE_MAPPED_SMILES, nconfs=0)
    angle = _angle(6, 3, 2, 491.0)  # matching N(H)-P-O angle below the floor
    fake_tfs = SimpleNamespace(uuid="x", mol_topos=[SimpleNamespace(angles=[angle])])
    _stub_tfs_build(monkeypatch, fake_tfs)

    result = ffparams_to_tfs(
        _ffparams(mol), _empty_bonded_data(), mol, mol_name="TEST", apply_hbond14_patch=False
    )

    assert result is fake_tfs
    assert angle.k == 491.0
