# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import shutil

import openmm as omm

from byteff2.md_utils.protocol import Protocol
from byteff2.toolkit.openmmtool import generate_byteffpol_system, nx_covalent_map_and_pairs


REPO_ROOT = Path(__file__).resolve().parents[3]
AFGBL_EXAMPLE_DIR = REPO_ROOT / "example/ByteFF-Pol/3_write_params/AFGBL"


def _build_afgbl_example_system(tmp_path: Path) -> tuple[Path, dict]:
    """Build the example single-component topology used by run_md.py."""
    params_dir = tmp_path / "params"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    params_dir.mkdir()
    output_dir.mkdir()

    for suffix in ("itp", "atp", "gro", "json"):
        shutil.copy(AFGBL_EXAMPLE_DIR / f"AFGBL.{suffix}", params_dir / f"AFGBL.{suffix}")

    protocol = Protocol(str(params_dir), str(output_dir))
    protocol.build_system(
        total_atoms=12,
        components_ratio={"AFGBL": 1},
        working_dir=str(work_dir),
        build_gas=True,
    )

    with open(params_dir / "AFGBL.json", "r") as f:
        nonbonded_params = {"AFGBL": json.load(f)}

    return params_dir / "system_gas.top", nonbonded_params


def _get_forces(system: omm.System) -> tuple[omm.AmoebaMultipoleForce, omm.CustomNonbondedForce, omm.CustomBondForce]:
    amoeba_force = None
    ljforce = None
    lj14force = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, omm.AmoebaMultipoleForce):
            amoeba_force = force
        elif isinstance(force, omm.CustomNonbondedForce):
            ljforce = force
        elif isinstance(force, omm.CustomBondForce):
            lj14force = force

    assert amoeba_force is not None
    assert ljforce is not None
    assert lj14force is not None
    return amoeba_force, ljforce, lj14force


def test_generate_byteffpol_system_consumes_example_write_params_output(tmp_path):
    top_file, nonbonded_params = _build_afgbl_example_system(tmp_path)

    top, system = generate_byteffpol_system(str(top_file), nonbonded_params)
    amoeba_force, ljforce, lj14force = _get_forces(system)

    residue = next(top.topology.residues())
    natoms = len(list(residue.atoms()))
    bonds = [[bond[0].index, bond[1].index] for bond in residue.bonds()]
    _, pairs = nx_covalent_map_and_pairs(natoms, bonds)

    assert system.getNumParticles() == natoms == len(nonbonded_params["AFGBL"]["charge"])
    assert amoeba_force.getNumMultipoles() == natoms
    assert ljforce.getNumParticles() == natoms
    assert ljforce.getNumExclusions() == sum(len(pairs[key]) for key in ("1-2", "1-3", "1-4", "1-5"))
    assert lj14force.getNumBonds() == len(pairs["1-4"]) + len(pairs["1-5"])
    assert amoeba_force.getNonbondedMethod() == omm.AmoebaMultipoleForce.NoCutoff
    assert ljforce.getNonbondedMethod() == omm.CustomNonbondedForce.NoCutoff


def test_generate_byteffpol_system_uses_periodic_methods_when_unit_cell_is_given(tmp_path):
    top_file, nonbonded_params = _build_afgbl_example_system(tmp_path)

    _, system = generate_byteffpol_system(
        str(top_file),
        nonbonded_params,
        unit_cell=[2.0, 2.0, 2.0],
        cutoff=1.0,
    )
    amoeba_force, ljforce, _ = _get_forces(system)

    assert amoeba_force.getNonbondedMethod() == omm.AmoebaMultipoleForce.PME
    assert ljforce.getNonbondedMethod() == omm.CustomNonbondedForce.CutoffPeriodic
    assert ljforce.getUseLongRangeCorrection() is True
    assert ljforce.getCutoffDistance().value_in_unit_system(omm.unit.md_unit_system) == 1.0
    assert amoeba_force.getCutoffDistance().value_in_unit_system(omm.unit.md_unit_system) == 1.0


# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
