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

"""Unit tests for protocol transport/build logic.

The tests in this module intentionally mock the heavy MD/OpenMM/Gromacs IO so
they can focus on the pure orchestration logic:

- ``TransportProtocol.post_process`` extracts species metadata and rounds
  near-integer molecular charges before calling ``onsager_calc``.
- ``Protocol.build_system`` orders components, checks charge neutrality, assigns
  molecule counts, and handles the gas-phase branch.
- conductivity helper regressions show why the rounded anion charge matters.
"""

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import torch

import byteff2.md_utils.onsager_conductivity as onsager_mod
from byteff2.md_utils.onsager_conductivity import (
    Dself_to_ionic_conductivity,
    Lambda_to_ionic_conductivity,
)


HEAVY_PROTOCOL_MODULES = (
    "byteff2.md_utils.md_run",
    "byteff2.md_utils.viscosity",
    "byteff2.toolkit.openmmtool",
    "byteff2.toolkit.hybridff_calculator",
    "openmm",
    "openmm.app",
    "openmm.unit",
)
MD_RUN_EXPORTS = (
    "dcd_read",
    "volume_calc",
    "viscosity_calc",
    "DipoleReporter",
    "npt_run",
    "nvt_run",
    "rescale_box",
)


def _make_post_component(name, charges, masses=None, molar_num=10):
    """Fake component shape consumed by ``TransportProtocol.post_process``."""
    masses = masses or [12.0] * len(charges)
    atoms = [SimpleNamespace(charge=charge, mass=mass) for charge, mass in zip(charges, masses)]
    return SimpleNamespace(name=name, atoms=atoms, molar_num=molar_num)


def _make_loaded_component(protocol_mod, name, net_charge, molar_ratio=1, n_atoms=3, atom_mass=12.0):
    """Fake component shape returned by ``load_topo`` for ``build_system``."""
    if net_charge > 1e-5:
        component_type = protocol_mod.ComponentType.CATION
        density = 0.25
    elif net_charge < -1e-5:
        component_type = protocol_mod.ComponentType.ANION
        density = 0.25
    else:
        component_type = protocol_mod.ComponentType.SOLVENT
        density = 0.9

    atoms = [SimpleNamespace(charge=net_charge / n_atoms, mass=atom_mass) for _ in range(n_atoms)]
    return SimpleNamespace(
        name=name,
        atoms=atoms,
        net_charge=net_charge,
        type=component_type,
        density=density,
        molar_ratio=molar_ratio,
        molar_num=-1,
        molar_mass=atom_mass * n_atoms,
        itp_records=SimpleNamespace(all=[]),
        atp_records=SimpleNamespace(all=[]),
    )


def _ordered_components(*items):
    return OrderedDict(items)


@contextmanager
def _noop_temporary_cd(_path):
    yield


@pytest.fixture(scope="module")
def protocol_mod():
    """Import protocol.py once with optional heavy dependencies mocked."""
    mocked_modules = {mod_name: MagicMock() for mod_name in HEAVY_PROTOCOL_MODULES if mod_name not in sys.modules}

    with patch.dict(sys.modules, mocked_modules):
        md_run = sys.modules.get("byteff2.md_utils.md_run", MagicMock())
        for export_name in MD_RUN_EXPORTS:
            setattr(md_run, export_name, MagicMock())

        import byteff2.md_utils.protocol as imported_protocol_mod

    return imported_protocol_mod


@pytest.fixture(scope="module")
def md_run_mod():
    fake_openmm = ModuleType("openmm")
    fake_openmm.System = object
    fake_openmm.Vec3 = object
    fake_openmm.Integrator = object
    fake_openmm.OpenMMException = RuntimeError
    fake_app = ModuleType("openmm.app")
    fake_app.StateDataReporter = object
    fake_app.Simulation = object
    fake_unit = ModuleType("openmm.unit")
    fake_gromacs = ModuleType("openmm.app.gromacstopfile")
    fake_gromacs.GromacsTopFile = object
    fake_dcd = ModuleType("MDAnalysis.lib.formats.libdcd")
    fake_dcd.DCDFile = object
    fake_utils = ModuleType("byteff2.bytemol.utils")
    fake_utils.temporary_cd = _noop_temporary_cd

    with patch.dict(
        sys.modules,
        {
            "openmm": fake_openmm,
            "openmm.app": fake_app,
            "openmm.unit": fake_unit,
            "openmm.app.gromacstopfile": fake_gromacs,
            "MDAnalysis.lib.formats.libdcd": fake_dcd,
            "byteff2.bytemol.utils": fake_utils,
        },
    ):
        sys.modules.pop("byteff2.md_utils.md_run", None)
        import byteff2.md_utils.md_run as imported_md_run

    return imported_md_run


@pytest.fixture(scope="module")
def viscosity_mod(md_run_mod):
    fake_openmm = ModuleType("openmm")
    fake_openmm.System = object
    fake_openmm.Vec3 = object
    fake_app = ModuleType("openmm.app")
    fake_app.Simulation = object
    fake_unit = ModuleType("openmm.unit")
    fake_gromacs = ModuleType("openmm.app.gromacstopfile")
    fake_gromacs.GromacsTopFile = object
    fake_utils = ModuleType("byteff2.bytemol.utils")
    fake_utils.temporary_cd = _noop_temporary_cd

    with patch.dict(
        sys.modules,
        {
            "openmm": fake_openmm,
            "openmm.app": fake_app,
            "openmm.unit": fake_unit,
            "openmm.app.gromacstopfile": fake_gromacs,
            "byteff2.bytemol.utils": fake_utils,
            "byteff2.md_utils.md_run": md_run_mod,
        },
    ):
        sys.modules.pop("byteff2.md_utils.viscosity", None)
        import byteff2.md_utils.viscosity as imported_viscosity

    return imported_viscosity


@pytest.fixture
def postprocess_env(protocol_mod, monkeypatch):
    """Patch ``TransportProtocol.post_process`` collaborators and expose mocks."""
    env = SimpleNamespace(
        protocol_mod=protocol_mod,
        onsager=MagicMock(return_value={"conductivity_onsager": 15.0}),
        viscosity=MagicMock(return_value=3.0),
        volume=MagicMock(return_value=(100000.0, 353.0)),
        dcd=MagicMock(return_value=np.zeros((500, 100, 3))),
        json=MagicMock(),
    )

    monkeypatch.setattr(protocol_mod, "onsager_calc", env.onsager)
    monkeypatch.setattr(protocol_mod, "viscosity_calc", env.viscosity)
    monkeypatch.setattr(protocol_mod, "volume_calc", env.volume)
    monkeypatch.setattr(protocol_mod, "dcd_read", env.dcd)
    monkeypatch.setattr(protocol_mod, "json", env.json)
    monkeypatch.setattr(protocol_mod.os, "makedirs", MagicMock())
    monkeypatch.setattr("builtins.open", mock_open())

    def run(components, onsager_result=None):
        if onsager_result is not None:
            env.onsager.return_value = onsager_result
        protocol = protocol_mod.TransportProtocol({"params_dir": "/tmp/p", "output_dir": "/tmp/o"})
        protocol.components = components
        protocol.post_process()
        return protocol

    env.run = run
    return env


@pytest.fixture
def build_system_env(protocol_mod, monkeypatch):
    """Patch ``Protocol.build_system`` collaborators and expose a compact runner."""
    env = SimpleNamespace(protocol_mod=protocol_mod)
    env.subprocess_run = MagicMock(return_value=MagicMock(returncode=0))
    env.topparse = MagicMock(
        molecules=[],
        strs_system_top_atp_itp=MagicMock(return_value=("system.top contents",)),
    )

    topo_full_system = MagicMock()
    topo_full_system.from_records.return_value = env.topparse

    monkeypatch.setattr(protocol_mod, "shutil", MagicMock())
    monkeypatch.setattr(protocol_mod, "predict_density", MagicMock(return_value=0.9))
    monkeypatch.setattr(protocol_mod, "predict_box", MagicMock(return_value=4.0))
    monkeypatch.setattr(protocol_mod, "generate_system_gro", MagicMock())
    monkeypatch.setattr(protocol_mod, "TopoFullSystem", topo_full_system)
    monkeypatch.setattr(protocol_mod, "RecordMolecule", MagicMock())
    monkeypatch.setattr(protocol_mod, "RecordAtomType", MagicMock())
    monkeypatch.setattr(protocol_mod, "subprocess", MagicMock(run=env.subprocess_run))
    monkeypatch.setattr(protocol_mod.os, "makedirs", MagicMock())

    def run(components, ratios, mix, total_atoms=300, build_gas=False, read_data="comment\n 100\n"):
        monkeypatch.setattr(protocol_mod, "load_topo", MagicMock(side_effect=lambda _dir, name: components[name]))
        monkeypatch.setattr(protocol_mod, "search_mixture", MagicMock(return_value=(total_atoms, np.array(mix))))
        monkeypatch.setattr("builtins.open", mock_open(read_data=read_data))

        protocol = protocol_mod.Protocol(params_dir="/tmp/p", output_dir="/tmp/o")
        return protocol.build_system(
            total_atoms=total_atoms,
            components_ratio=ratios,
            working_dir="/tmp/w",
            build_gas=build_gas,
        )

    env.run = run
    return env


class TestTransportPostProcess:
    """Focused tests for metadata extraction in ``post_process``."""

    def test_rounds_near_integer_charges_before_onsager(self, postprocess_env):
        postprocess_env.run(
            _ordered_components(
                ("TFSI", _make_post_component("TFSI", [-0.3333, -0.3333, -0.3333], molar_num=70)),
                ("LI", _make_post_component("LI", [1.0], molar_num=70)),
                ("SOL", _make_post_component("SOL", [0.5, -0.3, -0.2], molar_num=200)),
            )
        )

        assert postprocess_env.onsager.call_args[0][3] == {"TFSI": -1, "LI": 1, "SOL": 0}

    def test_passes_species_metadata_to_onsager(self, postprocess_env):
        postprocess_env.run(
            _ordered_components(
                (
                    "TFSI",
                    _make_post_component("TFSI", [-0.3333, -0.3333, -0.3333], masses=[32.0, 32.0, 32.0], molar_num=70),
                ),
                ("LI", _make_post_component("LI", [1.0], masses=[6.94], molar_num=70)),
            )
        )

        species_order, mass_dict, number_dict, charges_dict, volume, viscosity, temperature, _ = (
            postprocess_env.onsager.call_args[0]
        )
        assert species_order == ["TFSI", "LI"]
        assert mass_dict == {"TFSI": [32.0, 32.0, 32.0], "LI": [6.94]}
        assert number_dict == {"TFSI": 70, "LI": 70}
        assert charges_dict == {"TFSI": -1, "LI": 1}
        assert (volume, viscosity, temperature) == (100000.0, 3.0, 353.0)

    def test_writes_results_with_protocol_fields(self, postprocess_env):
        postprocess_env.run(
            _ordered_components(
                ("PF6", _make_post_component("PF6", [-1.0], molar_num=70)),
                ("LI", _make_post_component("LI", [1.0], molar_num=70)),
            ),
            onsager_result={"conductivity_onsager": 12.5},
        )

        dumped = postprocess_env.json.dump.call_args[0][0]
        assert dumped == {
            "conductivity_onsager": 12.5,
            "viscosity": 3.0,
            "components": ["PF6", "LI"],
        }

    def test_rejects_non_integer_total_charge(self, postprocess_env):
        with pytest.raises(AssertionError):
            postprocess_env.run(
                _ordered_components(
                    ("BAD", _make_post_component("BAD", [0.25, 0.25], molar_num=10)),
                )
            )


class TestBuildSystem:
    """Focused tests for ``Protocol.build_system`` orchestration."""

    def test_orders_components_and_assigns_molar_nums(self, protocol_mod, build_system_env):
        components = {
            "CAT": _make_loaded_component(protocol_mod, "CAT", net_charge=+1.0, molar_ratio=2),
            "SOL": _make_loaded_component(protocol_mod, "SOL", net_charge=0.0, molar_ratio=10),
            "ANI": _make_loaded_component(protocol_mod, "ANI", net_charge=-1.0, molar_ratio=2),
        }

        result = build_system_env.run(
            components=components,
            ratios={"CAT": 2, "SOL": 10, "ANI": 2},
            mix=[10, 5, 5],
        )

        assert list(result) == ["SOL", "ANI", "CAT"]
        assert [component.molar_num for component in result.values()] == [10, 5, 5]
        assert build_system_env.subprocess_run.called

    def test_rejects_charge_imbalanced_ratios_before_subprocess(self, protocol_mod, build_system_env):
        components = {
            "CAT": _make_loaded_component(protocol_mod, "CAT", net_charge=+1.0, molar_ratio=2),
            "ANI": _make_loaded_component(protocol_mod, "ANI", net_charge=-1.0, molar_ratio=1),
        }

        with pytest.raises(AssertionError, match="System charge"):
            build_system_env.run(
                components=components,
                ratios={"CAT": 2, "ANI": 1},
                mix=[5, 5],
            )

        assert not build_system_env.subprocess_run.called

    def test_gas_phase_single_component_returns_before_subprocess(self, protocol_mod, build_system_env):
        components = {
            "SOL": _make_loaded_component(protocol_mod, "SOL", net_charge=0.0, molar_ratio=1),
        }

        result = build_system_env.run(
            components=components,
            ratios={"SOL": 1},
            mix=[5],
            total_atoms=15,
            build_gas=True,
            read_data="title\n 5\nrest line\n 4.0 4.0 4.0\n",
        )

        assert list(result) == ["SOL"]
        assert not build_system_env.subprocess_run.called

    def test_gas_phase_rejects_multiple_components(self, protocol_mod, build_system_env):
        components = {
            "A": _make_loaded_component(protocol_mod, "A", net_charge=0.0),
            "B": _make_loaded_component(protocol_mod, "B", net_charge=0.0),
        }

        with pytest.raises(AssertionError, match="Gas phase"):
            build_system_env.run(
                components=components,
                ratios={"A": 1, "B": 1},
                mix=[5, 5],
                total_atoms=30,
                build_gas=True,
            )


class TestConductivityChargeRegression:
    """Regression tests showing why anion charge rounding matters."""

    LAMBDA_MD = torch.tensor(
        [
            [0.19791634414560155, 0.014926545960840962, -0.10181786258764798, -0.17483427265013932],
            [0.014926545960840962, 0.18525134374888982, -0.030071791904570693, -0.003072054487978046],
            [-0.10181786258764798, -0.030071791904570693, 0.7859389320018438, -0.6421221631313574],
            [-0.17483427265013932, -0.003072054487978046, -0.6421221631313574, 0.885835162382996],
        ],
        dtype=torch.float64,
    )
    DSELF_MD = torch.tensor(
        [3.7739033106591457, 2.856592183052462, 4.1247878411955385, 3.8609787802987743],
        dtype=torch.float64,
    )
    COUNTS = torch.tensor([70, 70, 238, 558], dtype=torch.float64)
    T = 353.0
    V = 115717.82
    N = 70 + 70 + 238 + 558
    CORRECT_CHARGES = torch.tensor([-1, 1, 0, 0], dtype=torch.float64)
    WRONG_CHARGES = torch.tensor([0, 1, 0, 0], dtype=torch.float64)

    def _onsager_sigma(self, charges):
        return Lambda_to_ionic_conductivity(self.LAMBDA_MD, charges, self.N, self.T, self.V).item()

    def _ne_sigma(self, charges):
        return Dself_to_ionic_conductivity(self.DSELF_MD, charges, self.COUNTS, self.T, self.V).item()

    def test_correct_charges_match_expected_conductivity_range(self):
        assert 14.0 < self._onsager_sigma(self.CORRECT_CHARGES) < 17.0
        assert 19.0 < self._ne_sigma(self.CORRECT_CHARGES) < 24.0

    def test_wrong_zero_anion_charge_changes_conductivity_significantly(self):
        onsager_delta = abs(self._onsager_sigma(self.CORRECT_CHARGES) - self._onsager_sigma(self.WRONG_CHARGES))
        ne_ratio = self._ne_sigma(self.WRONG_CHARGES) / self._ne_sigma(self.CORRECT_CHARGES)

        assert onsager_delta > 5.0
        assert ne_ratio < 0.6


class TestProtocolHelpers:
    def test_component_classifies_charge_and_density(self, protocol_mod):
        neutral = protocol_mod.Component(
            SimpleNamespace(
                name="SOL", atoms=[SimpleNamespace(charge=0.2, mass=12.0), SimpleNamespace(charge=-0.2, mass=1.0)]
            )
        )
        cation = protocol_mod.Component(SimpleNamespace(name="LI", atoms=[SimpleNamespace(charge=1.0, mass=6.94)]))
        anion = protocol_mod.Component(SimpleNamespace(name="PF6", atoms=[SimpleNamespace(charge=-1.0, mass=145.0)]))

        assert neutral.type is protocol_mod.ComponentType.SOLVENT
        assert neutral.density == pytest.approx(0.9)
        assert cation.type is protocol_mod.ComponentType.CATION
        assert anion.type is protocol_mod.ComponentType.ANION
        assert cation.molar_mass == pytest.approx(6.94)

    def test_predict_density_search_mixture_and_predict_box(self, protocol_mod):
        components = {
            "SOL": SimpleNamespace(
                type=protocol_mod.ComponentType.SOLVENT, density=0.9, molar_num=10, molar_mass=18.0, atoms=[1] * 3
            ),
            "CAT": SimpleNamespace(
                type=protocol_mod.ComponentType.CATION, density=0.25, molar_num=2, molar_mass=7.0, atoms=[1]
            ),
            "ANI": SimpleNamespace(
                type=protocol_mod.ComponentType.ANION, density=0.25, molar_num=2, molar_mass=145.0, atoms=[1] * 7
            ),
        }

        assert protocol_mod.predict_density(components) == pytest.approx(0.99)
        total_atoms, mix = protocol_mod.search_mixture(
            np.array([2, 4]), 50, {"A": SimpleNamespace(atoms=[1, 2]), "B": SimpleNamespace(atoms=[1, 2, 3])}
        )
        assert total_atoms >= 50
        np.testing.assert_array_equal(mix, np.array([7, 14]))
        assert protocol_mod.predict_box(components, density=1.0) == pytest.approx(
            round(((10 * 18 + 2 * 7 + 2 * 145) / 1.0) ** (1 / 3) * 0.11842, 2)
        )

    def test_load_topo_generate_system_gro_and_write_gro(self, protocol_mod, monkeypatch, tmp_path):
        itp_records = SimpleNamespace(all=["itp"])
        atp_records = SimpleNamespace(all=["atp"])
        topo_mol = SimpleNamespace(name="SOL", atoms=[SimpleNamespace(charge=0.0, mass=18.0)])
        monkeypatch.setattr(
            protocol_mod, "Records", SimpleNamespace(from_file=MagicMock(side_effect=[itp_records, atp_records]))
        )
        monkeypatch.setattr(
            protocol_mod,
            "TopoFullSystem",
            SimpleNamespace(from_records=MagicMock(return_value=SimpleNamespace(mol_topos=[topo_mol]))),
        )
        component = protocol_mod.load_topo(str(tmp_path), "SOL")
        assert component.name == "SOL"
        assert component.itp_records is itp_records
        assert component.atp_records is atp_records

        events = []

        class FakeScript:
            def add(self, cmd):
                events.append(("add", cmd))

            def init_gro_box(self, fp, box):
                events.append(("init", fp, box))

            def insert_molecules(self, fp, nmol):
                events.append(("insert", fp, nmol))

            def finish(self):
                events.append(("finish",))

            def write(self, fp):
                events.append(("write", fp))

        monkeypatch.setattr(protocol_mod, "GMXScript", FakeScript)
        comps = OrderedDict(
            [
                ("SOL", SimpleNamespace(name="SOL", molar_num=3)),
                ("CAT", SimpleNamespace(name="CAT", molar_num=2)),
            ]
        )
        protocol_mod.generate_system_gro(comps, str(tmp_path), 4.5)
        assert events[0][0] == "add"
        assert ("init", "SOL.gro", 4.5) in events
        assert ("insert", "SOL.gro", 2) in events
        assert ("insert", "CAT.gro", 2) in events

        atoms = SimpleNamespace(set_array=MagicMock())
        mol = SimpleNamespace(name="SOL", natoms=2, conformers=[SimpleNamespace(to_ase_atoms=lambda: atoms)])
        monkeypatch.setattr(protocol_mod.aio, "write", MagicMock())
        protocol_mod.write_gro(mol, str(tmp_path / "a.gro"))
        atoms.set_array.assert_called_once()
        protocol_mod.aio.write.assert_called_once()

    def test_generate_ff_params_writes_outputs(self, protocol_mod, monkeypatch, tmp_path):
        params_dir = tmp_path / "params"
        output_dir = tmp_path / "out"
        protocol = protocol_mod.Protocol(str(params_dir), str(output_dir))
        fake_model = object()
        fake_tfs = SimpleNamespace(write_itp=MagicMock())
        fake_mol = SimpleNamespace(name="SOL")

        monkeypatch.setattr(protocol_mod, "load_pretrained_model", lambda *_args, **_kwargs: fake_model)
        monkeypatch.setattr(protocol_mod, "write_gro", MagicMock())
        monkeypatch.setattr(
            protocol_mod,
            "Molecule",
            SimpleNamespace(from_smiles=lambda smiles, nconfs=1: SimpleNamespace(name="", smiles=smiles)),
        )
        monkeypatch.setattr(
            protocol_mod,
            "get_nb_params",
            lambda model, mol, write_to_itp=False: ({"source": "stub"}, {"charge": [0.1]}, fake_tfs, fake_mol),
        )

        result = protocol.generate_ff_params({"SOL": "O"})

        assert result == {"SOL": {"charge": [0.1]}, "metadata": {"source": "stub"}}
        fake_tfs.write_itp.assert_called_once_with(f"{params_dir}/SOL.itp", separated_atp=True)
        protocol_mod.write_gro.assert_called_once()
        assert (params_dir / "SOL.json").is_file()
        assert (params_dir / "SOL_nb_params.json").is_file()


class TestProtocolRunsAndPostprocess:
    def test_density_protocol_run_and_post_process(self, protocol_mod, monkeypatch, tmp_path):
        config = {
            "params_dir": str(tmp_path / "params"),
            "output_dir": str(tmp_path / "out"),
            "smiles": {"SOL": "O"},
            "natoms": 20,
            "components": {"SOL": 1},
            "working_dir": str(tmp_path / "work"),
            "temperature": 300.0,
        }
        protocol = protocol_mod.DensityProtocol(config)
        monkeypatch.setattr(protocol, "generate_ff_params", lambda smiles: {"SOL": {"charge": [0.1]}})
        monkeypatch.setattr(protocol, "build_system", lambda *args, **kwargs: {"SOL": 1})
        gro_parser = SimpleNamespace(positions="pos", getUnitCellDimensions=lambda: "box")
        monkeypatch.setattr(protocol_mod.app, "GromacsGroFile", lambda _fp: gro_parser)
        monkeypatch.setattr(protocol_mod, "generate_byteffpol_system", lambda *args, **kwargs: ("top", "system"))
        npt = MagicMock()
        monkeypatch.setattr(protocol_mod, "npt_run", npt)

        protocol.run_protocol()
        npt.assert_called_once()

        outdir = Path(config["output_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Density (g/mL)": np.linspace(1.0, 2.0, 1200)}).to_csv(outdir / "npt_state.csv", index=False)
        monkeypatch.setattr(np.random, "choice", lambda arr, size: np.asarray(arr)[:size])
        result = protocol.post_process()
        assert result["density"] == pytest.approx(np.mean(np.linspace(1.1668056713928275, 1.2493744787322769, 100)))
        assert (outdir / "density_results.json").is_file()

    def test_hvap_protocol_run_and_post_process(self, protocol_mod, monkeypatch, tmp_path):
        config = {
            "params_dir": str(tmp_path / "params"),
            "output_dir": str(tmp_path / "out"),
            "smiles": {"SOL": "O"},
            "natoms": 20,
            "components": {"SOL": 1},
            "working_dir": str(tmp_path / "work"),
            "temperature": 300.0,
        }
        protocol = protocol_mod.HVapProtocol(config)
        monkeypatch.setattr(protocol, "generate_ff_params", lambda smiles: {"SOL": {}})
        build_calls = []
        protocol.components = OrderedDict([("SOL", SimpleNamespace(molar_num=5))])
        monkeypatch.setattr(
            protocol,
            "build_system",
            lambda *args, **kwargs: build_calls.append(kwargs.get("build_gas", False)) or protocol.components,
        )
        gro_parser = SimpleNamespace(positions="pos", getUnitCellDimensions=lambda: "box")
        monkeypatch.setattr(protocol_mod.app, "GromacsGroFile", lambda _fp: gro_parser)
        monkeypatch.setattr(protocol_mod, "generate_byteffpol_system", lambda *args, **kwargs: ("top", "system"))
        monkeypatch.setattr(protocol_mod, "npt_run", MagicMock())
        monkeypatch.setattr(protocol_mod, "nvt_run", MagicMock())

        protocol.run_protocol()
        assert build_calls == [False, True]

        outdir = Path(config["output_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "Density (g/mL)": np.linspace(1.0, 1.5, 1200),
                "Potential Energy (kJ/mole)": np.linspace(-100.0, -80.0, 1200),
            }
        ).to_csv(outdir / "npt_state.csv", index=False)
        pd.DataFrame({"Potential Energy (kJ/mole)": np.linspace(-10.0, 0.0, 3000)}).to_csv(
            outdir / "nvt_state.csv", index=False
        )
        monkeypatch.setattr(np.random, "choice", lambda arr, size: np.asarray(arr)[:size])
        result = protocol.post_process()
        assert "hvap" in result and "density" in result
        assert (outdir / "hvap_results.json").is_file()

    def test_dielectric_protocol_run_and_post_process(self, protocol_mod, monkeypatch, tmp_path):
        config = {
            "params_dir": str(tmp_path / "params"),
            "output_dir": str(tmp_path / "out"),
            "smiles": {"SOL": "O"},
            "natoms": 20,
            "components": {"SOL": 1},
            "working_dir": str(tmp_path / "work"),
            "temperature": 300.0,
            "npt_steps": 10,
            "nvt_steps": 20,
            "dipole_interval": 5,
        }
        protocol = protocol_mod.DielectricProtocol(config)
        protocol.components = {"SOL": 1}
        monkeypatch.setattr(protocol, "generate_ff_params", lambda smiles: {"SOL": {}})
        monkeypatch.setattr(protocol, "build_system", lambda *args, **kwargs: {"SOL": 1})
        gro_parser = SimpleNamespace(positions="pos", getUnitCellDimensions=lambda: "box")
        monkeypatch.setattr(protocol_mod.app, "GromacsGroFile", lambda _fp: gro_parser)
        monkeypatch.setattr(protocol_mod, "generate_byteffpol_system", lambda *args, **kwargs: ("top", "system"))
        monkeypatch.setattr(protocol_mod, "npt_run", lambda *args, **kwargs: ("npt_pos", "npt_box"))
        monkeypatch.setattr(protocol_mod, "rescale_box", lambda *args, **kwargs: ("rs_pos", "rs_box"))

        captured = {}

        class FakeDipoleReporter:
            def __init__(self, file_path, reportInterval, system):
                captured.update({"file_path": file_path, "interval": reportInterval, "system": system})

        monkeypatch.setattr(protocol_mod, "DipoleReporter", FakeDipoleReporter)
        monkeypatch.setattr(protocol_mod, "nvt_run", MagicMock(return_value=("nvt_pos", "nvt_box")))

        protocol.run_protocol()
        assert captured["interval"] == 5

        outdir = Path(config["output_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "Mx_eA": np.linspace(1.0, 2.0, 20),
                "My_eA": np.linspace(0.5, 1.5, 20),
                "Mz_eA": np.linspace(0.1, 1.1, 20),
            }
        )
        df.to_csv(outdir / "dipole.csv", index=False)
        monkeypatch.setattr(protocol_mod, "volume_calc", lambda _wd: (5000.0, 300.0))
        result = protocol.post_process()
        assert result["units"]["correlation_time"] == "ps"
        assert (outdir / "dielectric_results.json").is_file()

    def test_compressibility_protocol_run_and_post_process(self, protocol_mod, monkeypatch, tmp_path):
        config = {
            "params_dir": str(tmp_path / "params"),
            "output_dir": str(tmp_path / "out"),
            "smiles": {"SOL": "O"},
            "natoms": 20,
            "components": {"SOL": 1},
            "working_dir": str(tmp_path / "work"),
            "temperature": 300.0,
            "npt_steps": 1000001,
        }
        protocol = protocol_mod.CompressibilityProtocol(config)
        monkeypatch.setattr(protocol, "generate_ff_params", lambda smiles: {"SOL": {}})
        monkeypatch.setattr(protocol, "build_system", lambda *args, **kwargs: {"SOL": 1})
        gro_parser = SimpleNamespace(positions="pos", getUnitCellDimensions=lambda: "box")
        monkeypatch.setattr(protocol_mod.app, "GromacsGroFile", lambda _fp: gro_parser)
        monkeypatch.setattr(protocol_mod, "generate_byteffpol_system", lambda *args, **kwargs: ("top", "system"))
        npt = MagicMock(return_value=("pos", "box"))
        monkeypatch.setattr(protocol_mod, "npt_run", npt)
        protocol.run_protocol()
        npt.assert_called_once()

        outdir = Path(config["output_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Box Volume (nm^3)": np.linspace(1.0, 1.1, 2005)}).to_csv(outdir / "npt_state.csv", index=False)
        result = protocol.post_process()
        assert result["units"]["compressibility"] == "GPa^-1"
        assert (outdir / "compressibility_results.json").is_file()


class TestMdRunModule:
    def test_openmm_run_rescale_volume_dcd_and_wrappers(self, md_run_mod, monkeypatch, tmp_path):
        class ForceBase:
            def __init__(self, name):
                self._name = name
                self._group = None

            def setForceGroup(self, group):
                self._group = group

            def getForceGroup(self):
                return self._group

            def getName(self):
                return self._name

        class AmoebaForce(ForceBase):
            pass

        class NonbondedForce(ForceBase):
            pass

        class CustomNonbondedForce(ForceBase):
            pass

        class BondForce(ForceBase):
            pass

        class FakePlatform:
            @staticmethod
            def getPlatformByName(_name):
                return SimpleNamespace(setPropertyDefaultValue=MagicMock())

        class FakeContext:
            def __init__(self):
                self.positions = None
                self.box = None
                self.temperature = None

            def setPositions(self, positions):
                self.positions = positions

            def setPeriodicBoxVectors(self, *box_vec):
                self.box = box_vec

            def setVelocitiesToTemperature(self, temperature):
                self.temperature = temperature

            def getState(self, **_kwargs):
                return SimpleNamespace(
                    getPositions=lambda: "final_positions",
                    getPeriodicBoxVectors=lambda: "final_box",
                )

        class FakeSimulation:
            def __init__(self, *_args):
                self.context = FakeContext()
                self.reporters = []
                self.currentStep = 2

            def minimizeEnergy(self, **_kwargs):
                self.minimized = True

            def step(self, steps):
                self.currentStep += steps

        fake_omm = SimpleNamespace(
            AmoebaMultipoleForce=AmoebaForce,
            NonbondedForce=NonbondedForce,
            CustomNonbondedForce=CustomNonbondedForce,
            Platform=FakePlatform,
            MonteCarloBarostat=lambda *args: ("barostat", args),
            MTSLangevinIntegrator=lambda *args: ("integrator", args),
            Vec3=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z, __mul__=lambda self, other: self),
        )
        fake_unit = SimpleNamespace(
            kilojoules_per_mole=1,
            nanometer=1,
            kelvin=1,
            picosecond=1,
            femtoseconds=1,
            atmospheres=1,
            nanometers=1,
        )
        fake_app = SimpleNamespace(
            Simulation=FakeSimulation,
            StateDataReporter=lambda **kwargs: ("state", kwargs),
            DCDReporter=lambda *args, **kwargs: ("dcd", args, kwargs),
        )
        monkeypatch.setattr(md_run_mod, "omm", fake_omm)
        monkeypatch.setattr(md_run_mod, "ou", fake_unit)
        monkeypatch.setattr(md_run_mod, "app", fake_app)

        class FakeSystem:
            def __init__(self):
                self.forces = [AmoebaForce("amoeba"), NonbondedForce("nb"), BondForce("bond")]

            def getNumForces(self):
                return len(self.forces)

            def getForce(self, idx):
                return self.forces[idx]

            def addForce(self, force):
                self.added = force

        positions, box = md_run_mod.openmm_run(
            task_name="demo",
            top=SimpleNamespace(topology="top"),
            system=FakeSystem(),
            positions="init_pos",
            integrator="integ",
            reporter="rep",
            work_dir=str(tmp_path),
            minimize=True,
            box_vec=(1, 2, 3),
            steps=10,
            temperature=300.0,
        )
        assert (positions, box) == ("final_positions", "final_box")

        monkeypatch.setattr(
            md_run_mod.pd, "read_csv", lambda _fp: pd.DataFrame({"Box Volume (nm^3)": np.linspace(8.0, 27.0, 600)})
        )

        class Vec:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

            def __mul__(self, _other):
                return self

        def vec_factory(x, y, z):
            return Vec(x, y, z)

        monkeypatch.setattr(md_run_mod.omm, "Vec3", vec_factory)
        scaled_pos, scaled_box = md_run_mod.rescale_box(
            np.array([[1.0, 2.0, 3.0]]),
            [SimpleNamespace(x=2.0, y=0.0, z=0.0), SimpleNamespace(x=0.0, y=2.0, z=0.0)],
            work_dir=str(tmp_path),
        )
        assert scaled_pos.shape == (1, 3)
        assert len(scaled_box) == 2

        monkeypatch.setattr(
            md_run_mod.pd,
            "read_csv",
            lambda _fp: pd.DataFrame({"Box Volume (nm^3)": [1.0, 2.0], "Temperature (K)": [300.0, 310.0]}),
        )
        volume, temp = md_run_mod.volume_calc(str(tmp_path))
        assert (volume, temp) == (1500.0, 305.0)

        class FakeDCD:
            def __enter__(self):
                return iter(
                    [SimpleNamespace(xyz=np.array([[1.0, 2.0, 3.0]])), SimpleNamespace(xyz=np.array([[4.0, 5.0, 6.0]]))]
                )

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(md_run_mod, "DCDFile", lambda _fp: FakeDCD())
        arr = md_run_mod.dcd_read("demo.dcd")
        assert arr.shape == (2, 1, 3)

        called = []
        monkeypatch.setattr(md_run_mod, "openmm_run", lambda **kwargs: called.append(kwargs) or ("p", "b"))
        monkeypatch.setattr(md_run_mod, "copy", SimpleNamespace(deepcopy=lambda x: x))
        md_run_mod.npt_run(
            top="top", system=FakeSystem(), positions="pos", npt_steps=20, temperature=300.0, work_dir=str(tmp_path)
        )
        md_run_mod.nvt_run(
            top="top",
            system=FakeSystem(),
            positions="pos",
            box_vec="box",
            temperature=300.0,
            work_dir=str(tmp_path),
            nvt_steps=30,
            extra_reporters=["x"],
            minimize=True,
        )
        assert called[0]["task_name"] == "npt"
        assert called[1]["task_name"] == "nvt"
        assert called[1]["reporter"][-1] == "x"

    def test_dipole_reporter_handles_reinit_and_writes_row(self, md_run_mod, monkeypatch, tmp_path):
        class FakeValue:
            def __init__(self, value):
                self.value = value

            def value_in_unit(self, _unit):
                return self.value

        class FakeAmoebaForce:
            def __init__(self):
                self.calls = 0

            def getMultipoleParameters(self, idx):
                return [FakeValue(idx + 1.0)]

            def getInducedDipoles(self, _context):
                self.calls += 1
                if self.calls == 1:
                    raise md_run_mod.omm.OpenMMException("rebuild")
                return np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])

        fake_force = FakeAmoebaForce()
        monkeypatch.setattr(
            md_run_mod, "omm", SimpleNamespace(AmoebaMultipoleForce=FakeAmoebaForce, OpenMMException=RuntimeError)
        )
        monkeypatch.setattr(md_run_mod, "ou", SimpleNamespace(elementary_charge=1, picoseconds=1, angstrom=1))
        system = SimpleNamespace(getNumForces=lambda: 1, getForce=lambda _idx: fake_force, getNumParticles=lambda: 2)
        reporter = md_run_mod.DipoleReporter(str(tmp_path / "dipole.csv"), 5, system)
        desc = reporter.describeNextReport(SimpleNamespace(currentStep=7))
        assert desc == {"steps": 3, "periodic": False, "include": ["positions"]}
        state = SimpleNamespace(
            getTime=lambda: FakeValue(1.5),
            getPositions=lambda asNumpy=True: FakeValue(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])),
        )
        simulation = SimpleNamespace(context=object(), system=system)
        reporter.report(simulation, state)
        text = Path(tmp_path / "dipole.csv").read_text(encoding="utf-8")
        assert "time_ps" in text and "1.5000" in text


class TestViscosityModule:
    def test_viscosity_reporter_nonequ_run_and_calc(self, viscosity_mod, monkeypatch, tmp_path):
        out = mock_open()
        monkeypatch.setattr("builtins.open", out)
        viscosity_mod.ou = SimpleNamespace(picosecond=1, nanometer=1, pascal=1, second=1, kelvin=1, femtoseconds=1)
        reporter = viscosity_mod.ViscosityReporter(str(tmp_path / "vis.csv"), 5)
        assert reporter.describeNextReport(SimpleNamespace(currentStep=6)) == {
            "steps": 4,
            "periodic": False,
            "include": [],
        }
        integ = SimpleNamespace(
            getCosAcceleration=lambda: SimpleNamespace(value_in_unit=lambda _unit: 0.2),
            getViscosity=lambda: (
                SimpleNamespace(value_in_unit=lambda _unit: 1.5),
                SimpleNamespace(value_in_unit=lambda _unit: 0.25),
            ),
        )
        reporter.report(SimpleNamespace(currentStep=10, integrator=integ), None)
        handle = out()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        assert "Acceleration" in written and "10\t0.2\t1.5\t0.25" in written

        monkeypatch.setattr(viscosity_mod, "copy", SimpleNamespace(deepcopy=lambda x: x))
        fake_integrator = SimpleNamespace(
            setUseMiddleScheme=MagicMock(),
            setCosAcceleration=MagicMock(),
        )
        fake_vv_mod = ModuleType("velocityverletplugin")
        fake_vv_mod.VVIntegrator = lambda **kwargs: fake_integrator
        monkeypatch.setitem(sys.modules, "velocityverletplugin", fake_vv_mod)
        called = []
        monkeypatch.setattr(viscosity_mod, "openmm_run", lambda **kwargs: called.append(kwargs) or ("p", "b"))
        monkeypatch.setattr(viscosity_mod, "omm", SimpleNamespace())
        viscosity_mod.ou = SimpleNamespace(kelvin=1, picoseconds=1, femtoseconds=1)
        monkeypatch.setattr(viscosity_mod, "ViscosityReporter", lambda file, reportInterval: (file, reportInterval))
        viscosity_mod.nonequ_run(
            top="top",
            system="sys",
            positions="pos",
            box_vec="box",
            temperature=300.0,
            work_dir=str(tmp_path),
            nonequ_steps=50,
        )
        assert called[0]["task_name"] == "nonequ"

        monkeypatch.setattr(
            viscosity_mod.pd,
            "read_csv",
            lambda _fp, sep="\t": pd.DataFrame({"1/Viscosity (1/Pa.s)": np.ones(10050) * 0.5}),
        )
        assert viscosity_mod.viscosity_calc(str(tmp_path)) == pytest.approx(2000.0)


class TestOnsagerHelpers:
    def test_correlate_polyfit_indices_and_matrix_helpers(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
        corr = onsager_mod.correlate_xy(a, b)
        assert corr.shape == (3,)
        corr2 = onsager_mod.correlate_xy(torch.stack([a, b], dim=1), torch.stack([b, a], dim=1))
        assert corr2.shape == (3,)
        sol = onsager_mod.polyfit(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 3.0, 5.0]), 1)
        assert sol.shape == (2, 1)
        assert onsager_mod.build_Delta_index(3)[1] == 4
        assert onsager_mod.build_B_index(3)[1] == 4

        L = torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64)
        masses = torch.tensor([1.0, 2.0], dtype=torch.float64)
        corrected = onsager_mod.remove_center_of_mass_error(L, masses)
        assert corrected.shape == (2, 2)

        xfrac = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float64)
        delta, inv_delta = onsager_mod.Delta_matrices(torch.eye(3, dtype=torch.float64), xfrac)
        assert delta.shape == (2, 2)
        assert inv_delta.shape == (2, 2)
        ms = onsager_mod.MS_matrix(torch.eye(2, dtype=torch.float64), xfrac)
        assert ms.shape == (3, 3)
        xmat, xinv = onsager_mod.X_matrices(xfrac, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert xmat.shape == (4, 4)
        assert xinv.shape == (4, 4)
        assert onsager_mod.fsc_self_diffusivity(300.0, 1.0, 10.0) > 0
        full = onsager_mod.fsc_full_Lambda(0.5, xinv, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert full.shape == (3, 3)

    def test_onsager_calc_runs_with_mocked_correlations(self, monkeypatch):
        monkeypatch.setattr(
            onsager_mod, "correlate_xy", lambda *_args, **_kwargs: torch.linspace(0.0, 1.0, 205, dtype=torch.float64)
        )
        monkeypatch.setattr(
            onsager_mod, "polyfit", lambda *_args, **_kwargs: torch.tensor([[0.0], [2.0]], dtype=torch.float64)
        )
        positions = np.zeros((405, 2, 3), dtype=np.float64)
        result = onsager_mod.onsager_calc(
            species_order=["A", "B"],
            species_mass={"A": [1.0], "B": [2.0]},
            species_number={"A": 1, "B": 1},
            species_charge={"A": 1, "B": -1},
            volume_angstrom3=1000.0,
            viscosity_cP=1.0,
            T_K=300.0,
            positions=positions,
        )
        assert "conductivity_onsager" in result
        assert "Dself_inf" in result
