#!/usr/bin/env python
"""Validate a fresh ByteFF2 installation end to end, in about a minute.

Run this right after cloning + installing, before committing hours of GPU time
to a real property calculation. It exercises every part of the stack that a
misconfigured machine can break, using a small reference trajectory shipped in
this directory so that the numbers are reproducible.

Three stages:

  [1] VV plugin      -- import AND instantiate VVIntegrator.
                        Catches the most common install failure: the OpenMM
                        velocity-Verlet plugin's SWIG wrapper has no
                        RPATH/RUNPATH, so it resolves libOpenMMVelocityVerlet.so
                        only through LD_LIBRARY_PATH. install.sh appends that
                        export to the *installing* user's ~/.bashrc, which a
                        non-interactive shell (nohup / cron / systemd / IDE)
                        never sources. Without this stage the failure would only
                        surface at the very end of a transport run, after ~12 h
                        of MD, because the import lives inside nonequ_run().

  [2] Model + system -- load trained_models/optimal.pt, regenerate force field
                        parameters for EC/DMC/MA/LI/PF6 from SMILES, compare
                        against the committed .itp/.json, then build the
                        polarizable OpenMM system from system.top.
                        Catches a broken torch/CUDA, a corrupt checkpoint, or
                        an OpenMM that cannot construct AmoebaMultipoleForce.

  [3] Post-analysis  -- run the real viscosity and Onsager conductivity
                        analysis on the reference trajectory and compare with
                        expected_results.json.
                        Catches numerical/dependency drift (torch, numpy,
                        MDAnalysis, scipy).

Usage:
    python run_installation_test.py             # all stages
    python run_installation_test.py --stage 1   # just the plugin check
    python run_installation_test.py --regenerate-expected   # maintainers only
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

# LD_LIBRARY_PATH is consumed by the dynamic loader at process start, so setting
# it from inside Python is too late -- re-exec once so the check below is a real
# test of the library being findable, not of this script's own environment.
_OPENMM_LIB = os.path.join(os.environ.get('OPENMM_DIR', '/usr/local/openmm'), 'lib')
if os.path.isdir(_OPENMM_LIB) and _OPENMM_LIB not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _OPENMM_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, REPO)

# pylint: disable=wrong-import-position
import argparse
import json
import tempfile
import traceback
from typing import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'reference')
PARAMS = os.path.join(REF, 'params')
EXPECTED = os.path.join(HERE, 'expected_results.json')

# Analysis is float64 and deterministic, but BLAS/hardware differences move the
# last few digits; 0.5% is far tighter than any physical significance here.
RTOL = 5e-3
# Model inference in float32 varies a little more across CPU/GPU builds.
PARAM_RTOL = 1e-3

# Files that stage 3 needs. Missing ones mean an incomplete checkout, not a
# broken installation, so they get their own message rather than a traceback.
REQUIRED_REFERENCE = {
    'nvt.dcd': 'reference NVT trajectory (600 frames)',
    'nvt_state.csv': 'box volume and temperature for those frames',
    'viscosity.csv': 'non-equilibrium shear output',
    'params/system.top': 'system topology',
}

SMILES = {
    'EC': 'O=C1OCCO1',
    'DMC': 'COC(=O)OC',
    'MA': 'COC(C)=O',
    'LI': '[Li+]',
    'PF6': 'F[P-](F)(F)(F)(F)F',
}


def banner(text):
    print(f'\n{"=" * 74}\n{text}\n{"=" * 74}', flush=True)


def ok(text):
    print(f'  [ OK ] {text}', flush=True)


def fail(text, hint=None):
    print(f'  [FAIL] {text}', flush=True)
    if hint:
        for line in hint.strip().splitlines():
            print(f'         {line}', flush=True)


def read_molecules(top_file):
    """Parse [ molecules ] preserving order (onsager_calc slices by this order)."""
    molecules, in_section = [], False
    with open(top_file) as f:
        for line in f:
            stripped = line.split(';')[0].strip()
            if not stripped:
                continue
            if stripped.startswith('['):
                in_section = stripped.strip('[] ').strip() == 'molecules'
                continue
            if in_section:
                name, num = stripped.split()
                molecules.append((name, int(num)))
    return molecules


def stage1_plugin():
    banner('[1/4] OpenMM velocity-Verlet plugin')
    hint = """
The plugin's shared library could not be loaded. Its SWIG wrapper has no
RPATH, so LD_LIBRARY_PATH must contain <OPENMM_DIR>/lib. Fix permanently with:

    echo /usr/local/openmm/lib | sudo tee /etc/ld.so.conf.d/openmm.conf
    sudo ldconfig

or per-shell:

    export LD_LIBRARY_PATH=/usr/local/openmm/lib:$LD_LIBRARY_PATH

If the library does not exist at all, (re)run submodules/openmm/install.sh.
"""
    try:
        import openmm as omm
        import openmm.unit as ou
        ok(f'openmm {omm.version.version}')
        print(f'         library path: {omm.version.openmm_library_path}', flush=True)

        from velocityverletplugin import VVIntegrator
        ok('import velocityverletplugin.VVIntegrator')

        # Importing is not enough -- instantiate, which touches the C++ library.
        integrator = VVIntegrator(298 * ou.kelvin, 1.0 / ou.picoseconds, 298 * ou.kelvin,
                                  100 / ou.picoseconds, 1 * ou.femtoseconds, 3, 3)
        integrator.setUseMiddleScheme(True)
        integrator.setCosAcceleration(0.02)
        ok(f'instantiate VVIntegrator (cosAcceleration={integrator.getCosAcceleration()})')
        return True
    except Exception as exc:  # pylint: disable=broad-except
        fail(f'{type(exc).__name__}: {exc}', hint)
        return False


def _compare_numeric_text(generated, committed, label, results):
    """Token-wise compare two .itp files, tolerating float noise and timestamps."""
    gen = [l for l in generated.splitlines() if not l.strip().startswith(';')]
    com = [l for l in committed.splitlines() if not l.strip().startswith(';')]
    if len(gen) != len(com):
        results.append(f'{label}: line count {len(gen)} != {len(com)}')
        return
    for lineno, (a, b) in enumerate(zip(gen, com), 1):
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            results.append(f'{label}:{lineno}: token count differs')
            return
        for x, y in zip(ta, tb):
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                if x != y:
                    results.append(f'{label}:{lineno}: {x!r} != {y!r}')
                    return
                continue
            if not np.isclose(fx, fy, rtol=PARAM_RTOL, atol=1e-6):
                results.append(f'{label}:{lineno}: {fx} != {fy}')
                return


def stage2_model():
    banner('[2/4] Model checkpoint, force field parameters, OpenMM system')
    try:
        from byteff2.toolkit.openmmtool import generate_openmm_system
        from byteff2.train.utils import get_nb_params, load_model
        from bytemol.core import Molecule
        from bytemol.utils import get_data_file_path
        import openmm as omm
        import openmm.unit as ou

        model_dir = os.path.dirname(get_data_file_path('trained_models/optimal.pt', 'byteff2'))
        model = load_model(model_dir)
        ok(f'loaded model from {model_dir}')

        mismatches = []
        for name, smiles in SMILES.items():
            mol = Molecule.from_smiles(smiles, nconfs=1)
            mol.name = name
            _metadata, params, tfs, _mol = get_nb_params(model, mol)

            # nonbonded: compare the 9 arrays against the committed JSON
            with open(os.path.join(PARAMS, f'{name}.json')) as f:
                reference = json.load(f)
            for key, ref_values in reference.items():
                got = np.asarray(params[key], dtype=float)
                exp = np.asarray(ref_values, dtype=float)
                if got.shape != exp.shape:
                    mismatches.append(f'{name}.{key}: shape {got.shape} != {exp.shape}')
                elif not np.allclose(got, exp, rtol=PARAM_RTOL, atol=1e-8):
                    worst = float(np.max(np.abs(got - exp) / (np.abs(exp) + 1e-12)))
                    mismatches.append(f'{name}.{key}: max rel dev {worst:.2e}')

            # bonded: regenerate the .itp and compare numerically
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, f'{name}.itp')
                tfs.write_itp(out, separated_atp=True)
                with open(out) as f:
                    generated = f.read()
            with open(os.path.join(PARAMS, f'{name}.itp')) as f:
                committed = f.read()
            _compare_numeric_text(generated, committed, f'{name}.itp', mismatches)

        if mismatches:
            for line in mismatches[:10]:
                fail(line)
            fail('regenerated parameters differ from the committed reference',
                 'A torch/CUDA difference or a corrupt optimal.pt checkpoint is the usual cause.')
            return False
        ok(f'regenerated parameters match reference for {", ".join(SMILES)}')

        # build the polarizable system exactly as the protocols do
        _top, system = build_reference_system()
        nparticles = system.getNumParticles()
        forces = sorted({system.getForce(i).getName() for i in range(system.getNumForces())})
        if nparticles != 4012:
            fail(f'system has {nparticles} particles, expected 4012')
            return False
        ok(f'built OpenMM system: {nparticles} particles')
        print(f'         forces: {", ".join(forces)}', flush=True)
        for required in ('AmoebaMultipoleForce', 'CustomNonbondedForce'):
            if required not in forces:
                fail(f'missing {required} -- the polarizable nonbonded rebuild did not happen')
                return False
        ok('polarizable nonbonded forces present')
        return True
    except Exception as exc:  # pylint: disable=broad-except
        fail(f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        return False


def compute_analysis():
    """Run the real post-MD analysis on the reference trajectory."""
    from byteff2.md_utils.md_run import dcd_read, volume_calc
    from byteff2.md_utils.onsager_conductivity import onsager_calc
    from byteff2.md_utils.viscosity import viscosity_calc
    from byteff2.toolkit.protocol import load_topo

    viscosity = viscosity_calc(REF)
    volume, temperature = volume_calc(REF)
    positions = dcd_read(os.path.join(REF, 'nvt.dcd'))

    components = OrderedDict()
    for name, num in read_molecules(os.path.join(PARAMS, 'system.top')):
        component = load_topo(PARAMS, name)
        component.molar_num = num
        components[name] = component

    species_mass, species_number, species_charge = {}, {}, {}
    for name, component in components.items():
        species_mass[name] = [atom.mass for atom in component.atoms]
        species_number[name] = component.molar_num
        species_charge[name] = int(round(sum(atom.charge for atom in component.atoms)))

    results = onsager_calc(list(components), species_mass, species_number, species_charge, volume, viscosity,
                           temperature, positions)
    results['viscosity'] = viscosity
    results['components'] = list(components)

    context = {
        'order': list(components),
        'species_mass': species_mass,
        'species_number': species_number,
        'species_charge': species_charge,
        'volume': volume,
        'temperature': temperature,
        'viscosity': viscosity,
        'positions': positions,
        'top_order': [name for name, _ in read_molecules(os.path.join(PARAMS, 'system.top'))],
    }
    return results, context


def verify_ionic_assignment(results, context):
    """Confirm conductivity uses only the ionic species.

    Both conductivity routes are quadratic in the charge vector --
    sigma = e^2 * z^T L z (Onsager) and e^2 * sum(z_i^2 D_i N_i) (Nernst-Einstein)
    -- so a species with z = 0 contributes exactly zero, diagonal and cross terms
    alike. Neutral solvents therefore cannot enter the conductivity *provided the
    charge vector lines up with the trajectory's per-species atom blocks*.

    onsager_calc never checks that alignment: it slices the trajectory using
    `species_order` but builds its charge vector from dict insertion order. If the
    two ever disagree, a neutral solvent silently gets an ionic charge. That is
    the failure this function guards against.
    """
    from byteff2.md_utils.onsager_conductivity import onsager_calc

    order = context['order']
    charge = context['species_charge']
    number = context['species_number']
    mass = context['species_mass']

    problems = []

    # 1. the order used to slice the trajectory must be the order the atoms are
    #    actually laid out in, i.e. system.top's [ molecules ] section
    if order != context['top_order']:
        problems.append(f'species order {order} != system.top [molecules] {context["top_order"]}')

    # 2. show the block layout and classification
    print('  species blocks in the trajectory:', flush=True)
    offset = 0
    ionic, neutral = [], []
    for name in order:
        natoms = len(mass[name]) * number[name]
        z = charge[name]
        kind = 'IONIC  ' if z else 'neutral'
        print(f'         {name:<5} atoms [{offset:5d}:{offset + natoms:5d}]  z={z:+d}  {kind}', flush=True)
        (ionic if z else neutral).append(name)
        offset += natoms

    # 3. charge neutrality of the whole box
    net = sum(charge[name] * number[name] for name in order)
    if net != 0:
        problems.append(f'net box charge is {net}, expected 0')

    # 4. there must be ions at all, otherwise conductivity is meaningless
    if not ionic:
        problems.append('no ionic species found -- conductivity would be identically zero')

    # 5. sensitivity proof: rolling the charges onto the wrong species must change
    #    the answer. If it does not, charges are being ignored somewhere and the
    #    neutral/ionic distinction is not actually being applied.
    rolled = dict(zip(order, [charge[name] for name in order[-1:] + order[:-1]]))
    perturbed = onsager_calc(order, mass, number, rolled, context['volume'], context['viscosity'],
                             context['temperature'], context['positions'])
    ref_sigma = results['conductivity_onsager']
    bad_sigma = perturbed['conductivity_onsager']
    if abs(bad_sigma - ref_sigma) <= 0.05 * abs(ref_sigma):
        problems.append('misassigning charges to neutral species did not change the conductivity; '
                        'the charge vector is not reaching the conductivity sums')

    if problems:
        for line in problems:
            fail(line)
        fail('ionic species are not being identified correctly',
             'Conductivity would be computed from the wrong molecules.')
        return False

    ok(f'ionic: {", ".join(ionic)}   neutral (excluded): {", ".join(neutral) or "none"}')
    ok(f'net box charge 0; species order matches system.top [molecules]')
    ok(f'charge-assignment'
       f'({abs(bad_sigma - ref_sigma) / abs(ref_sigma) * 100:.0f}% change, so charges do gate the sums)')
    return True


# The NVT box the reference trajectory was produced in (constant volume).
REFERENCE_BOX_NM = 3.55627


def build_reference_system():
    """Build the polarizable OpenMM system from the committed reference files."""
    from byteff2.toolkit.openmmtool import generate_openmm_system
    import openmm as omm
    import openmm.unit as ou

    nb_params = {}
    for name in SMILES:
        with open(os.path.join(PARAMS, f'{name}.json')) as f:
            nb_params[name] = json.load(f)
    with open(os.path.join(PARAMS, 'EC_nb_params.json')) as f:
        nb_params['metadata'] = json.load(f)['metadata']

    length = REFERENCE_BOX_NM
    unit_cell = omm.Vec3(length, length, length) * ou.nanometers
    return generate_openmm_system(os.path.join(PARAMS, 'system.top'), nb_params, unit_cell)


def stage4_nonequ():
    """Actually run nonequ_run briefly on the GPU.

    Stage 1 proves the plugin's shared library loads, but OpenMM integrators are
    platform-agnostic until a Context exists: the plugin's CUDA kernels are only
    compiled inside app.Simulation(...) (md_run.py:59). A machine where the
    plugin imports but its CUDA modules fail to build would sail through stage 1
    and still die at the end of a 12 h transport run.

    So call the real nonequ_run() for a few hundred steps. That exercises the
    whole path that fails: import, VVIntegrator construction, the Middle scheme
    and cosine-acceleration setters, Context creation and kernel compilation,
    stepping, integrator.getViscosity(), and ViscosityReporter's file output.

    Skipped (not failed) when no CUDA device is present, since openmm_run
    hardcodes the CUDA platform and every other protocol would fail first.
    """
    banner('[4/4] Short non-equilibrium (NEMD) run on the GPU')
    import openmm as omm

    try:
        omm.Platform.getPlatformByName('CUDA')
    except Exception:  # pylint: disable=broad-except
        print('  [SKIP] no CUDA platform available; nonequ requires it '
              '(openmm_run hardcodes CUDA)', flush=True)
        return None

    try:
        import warnings
        warnings.filterwarnings('ignore')
        import MDAnalysis as mda
        import openmm.unit as ou
        from byteff2.md_utils.viscosity import nonequ_run

        top, system = build_reference_system()
        ok('rebuilt reference system')

        # start from the last equilibrated NVT frame; nonequ_run does not minimize
        universe = mda.Universe(os.path.join(PARAMS, 'solvent_salt.gro'), os.path.join(REF, 'nvt.dcd'))
        universe.trajectory[-1]
        positions = ou.Quantity(value=[omm.Vec3(*xyz) for xyz in universe.atoms.positions / 10.0],
                                unit=ou.nanometer)
        length = REFERENCE_BOX_NM
        box_vec = [
            omm.Vec3(length, 0., 0.) * ou.nanometers,
            omm.Vec3(0., length, 0.) * ou.nanometers,
            omm.Vec3(0., 0., length) * ou.nanometers,
        ]

        steps = 200
        with tempfile.TemporaryDirectory() as tmp:
            nonequ_run(top, system, positions, box_vec, temperature=298.0, work_dir=tmp, nonequ_steps=steps)
            csv_path = os.path.join(tmp, 'viscosity.csv')
            if not os.path.exists(csv_path):
                fail('nonequ_run produced no viscosity.csv')
                return False
            import pandas as pd
            frame = pd.read_csv(csv_path, sep='\t')
            rows = len(frame)
            amplitude = frame['VelocityAmplitude (nm/ps)']
            inverse = frame['1/Viscosity (1/Pa.s)']

        ok(f'ran {steps} nonequ steps, ViscosityReporter wrote {rows} rows')
        if rows < 1:
            fail('ViscosityReporter wrote no data rows')
            return False
        if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(inverse)):
            fail('viscosity output contains NaN/inf -- the run is unstable')
            return False
        ok('integrator.getViscosity() returns finite values')
        # 1/eta must stay exactly proportional to the measured amplitude
        ratio = (inverse / amplitude).to_numpy()
        if not np.allclose(ratio, ratio[0], rtol=1e-6):
            fail('1/viscosity is not proportional to the velocity amplitude')
            return False
        ok('1/viscosity stays proportional to velocity amplitude (plugin math consistent)')
        return True
    except Exception as exc:  # pylint: disable=broad-except
        fail(f'{type(exc).__name__}: {exc}',
             'The plugin imports but its CUDA path is broken. Check that OpenMM and\n'
             'openmm-velocityVerlet were built against the same CUDA toolkit.')
        traceback.print_exc()
        return False


def check_reference_files(regenerate=False):
    """Report missing reference data clearly instead of failing deep in pandas."""
    missing = [(f'reference/{name}', what) for name, what in REQUIRED_REFERENCE.items()
               if not os.path.exists(os.path.join(REF, name))]
    if not regenerate and not os.path.exists(EXPECTED):
        missing.append((os.path.basename(EXPECTED), 'expected values to compare against'))
    if not missing:
        return True
    for name, what in missing:
        fail(f'missing {name}  -- {what}')
    fail('reference data is incomplete',
         'This is a checkout problem, not an installation problem.\n'
         'Re-clone or restore example/8_encapsulation/installation_test/reference/.\n'
         'Stages 1 and 2 above are still meaningful on their own.')
    return False


def stage3_analysis(regenerate=False):
    banner('[3/4] Post-MD analysis on the reference trajectory')
    if not check_reference_files(regenerate):
        return False
    try:
        results, context = compute_analysis()
        if not verify_ionic_assignment(results, context):
            return False
    except Exception as exc:  # pylint: disable=broad-except
        fail(f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        return False

    if regenerate:
        with open(EXPECTED, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        ok(f'wrote {EXPECTED}')
        return True

    with open(EXPECTED) as f:
        expected = json.load(f)

    bad = []
    for key, exp in expected.items():
        if key not in results:
            bad.append(f'{key}: missing from computed results')
            continue
        got = results[key]
        if isinstance(exp, (int, float)) and not isinstance(exp, bool):
            if not np.isclose(float(got), float(exp), rtol=RTOL, atol=1e-12):
                bad.append(f'{key}: {got!r} != {exp!r} (rtol={RTOL})')
            else:
                print(f'  [ OK ] {key:<28} {float(got):.6g}  (expected {float(exp):.6g})', flush=True)
        elif isinstance(exp, list) and exp and isinstance(exp[0], (int, float)):
            if not np.allclose(np.asarray(got, dtype=float), np.asarray(exp, dtype=float), rtol=RTOL, atol=1e-12):
                bad.append(f'{key}: array differs beyond rtol={RTOL}')
            else:
                print(f'  [ OK ] {key:<28} array[{len(exp)}]', flush=True)
        elif got != exp:
            bad.append(f'{key}: {got!r} != {exp!r}')

    if bad:
        for line in bad:
            fail(line)
        fail('analysis results drifted from the reference',
             'Usually a dependency version change (torch / numpy / MDAnalysis / scipy).\n'
             'Compare your versions against requirements.txt.')
        return False
    ok('all reference values reproduced')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], action='append',
                        help='run only this stage (repeatable); default is all')
    parser.add_argument('--regenerate-expected', action='store_true',
                        help='overwrite expected_results.json from this machine (maintainers only)')
    args = parser.parse_args()
    stages = sorted(set(args.stage)) if args.stage else [1, 2, 3, 4]

    print(f'ByteFF2 installation test\nrepo: {REPO}\npython: {sys.executable}')
    print(f'LD_LIBRARY_PATH: {os.environ.get("LD_LIBRARY_PATH", "(unset)")}')

    outcomes = {}
    if 1 in stages:
        outcomes['VV plugin'] = stage1_plugin()
    if 2 in stages:
        outcomes['model + system'] = stage2_model()
    if 3 in stages:
        outcomes['post-analysis'] = stage3_analysis(args.regenerate_expected)
    if 4 in stages and not args.regenerate_expected:
        result = stage4_nonequ()
        if result is not None:
            outcomes['nonequ (NEMD)'] = result

    banner('SUMMARY')
    for name, passed in outcomes.items():
        print(f'  {"PASS" if passed else "FAIL"}  {name}', flush=True)
    if all(outcomes.values()):
        print('\nInstallation looks good -- transport properties will run to completion.\n')
        return 0
    print('\nInstallation is NOT ready. Fix the failures above before running a protocol.\n')
    return 1


if __name__ == '__main__':
    sys.exit(main())
