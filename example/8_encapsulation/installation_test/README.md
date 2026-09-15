# Installation test

A ~1 minute end-to-end check that a freshly cloned + installed ByteFF2 will
actually run to completion. **Run this before starting any real property
calculation.**

A transport calculation costs roughly 12 hours of GPU time (4 ns NPT + 20 ns
NVT + 1 ns non-equilibrium shear). The most common installation problem —
a velocity-Verlet plugin that cannot load its shared library — does not surface
until the *last* of those three stages, because the import lives inside
`nonequ_run()`. You lose the whole run. This test reproduces that failure in
seconds instead.

## Quick start

```bash
cd example/8_encapsulation/installation_test
python run_installation_test.py
```

Exit code is `0` if everything passed, `1` otherwise.

```
  PASS  VV plugin
  PASS  model + system
  PASS  post-analysis
  PASS  nonequ (NEMD)

Installation looks good -- transport properties will run to completion.
```

Run a single stage with `--stage N` (repeatable), e.g. `--stage 1` for just the
plugin check.

## What each stage covers

| Stage | Checks | Needs GPU |
|---|---|---|
| **1. VV plugin** | `openmm` imports; `velocityverletplugin.VVIntegrator` imports **and instantiates** | no |
| **2. Model + system** | `optimal.pt` loads; parameters regenerated from SMILES for EC/DMC/MA/LI/PF6 match the committed `.itp`/`.json`; the polarizable OpenMM system builds from `system.top` with `AmoebaMultipoleForce` + `CustomNonbondedForce` present | no |
| **3. Post-analysis** | real `viscosity_calc` + `onsager_calc` on the reference trajectory, compared against `expected_results.json`; verifies that **only the ionic species enter the conductivity** | no |
| **4. NEMD run** | calls the real `nonequ_run()` for 200 steps: compiles the plugin's CUDA kernels, steps the integrator, reads `getViscosity()`, writes through `ViscosityReporter` | **yes** (skipped if absent) |

Stage 1 instantiates the integrator rather than only importing it, because the
import alone does not always touch the C++ library.

Stage 4 exists because stage 1 is not sufficient on its own. OpenMM integrators
are platform-agnostic until a `Context` exists — the plugin's CUDA kernels are
compiled inside `app.Simulation(...)` (`md_run.py:59`), which stage 1 never
reaches. You can see the difference in the output: only stage 4 prints

```
CUDA modules for velocity-Verlet-middle integrator created
CUDA modules for Nose-Hoover thermostat created
CUDA modules for CosineAccelerateModifier created
```

A machine where the plugin imports but its CUDA modules fail to build — OpenMM
and openmm-velocityVerlet compiled against different CUDA toolkits, say — passes
stage 1 and still dies at the end of a 12 hour transport run. Stage 4 calls the
real `nonequ_run()`, so it covers the entire failing path: import, VVIntegrator
construction, the Middle-scheme and cosine-acceleration setters, Context
creation and kernel compilation, stepping, `integrator.getViscosity()`, and
`ViscosityReporter` file output. It is skipped, not failed, when no CUDA device
is present, since `openmm_run` hardcodes the CUDA platform and every other
protocol would fail earlier anyway.

Stage 2 is what catches a broken torch/CUDA, a corrupt checkpoint, or an OpenMM
that cannot construct the polarizable forces. Note ByteFF2 predicts the
*whole* force field — bonded terms included — so this compares both the nine
nonbonded arrays and the regenerated `.itp` bonded blocks.

## Are the ionic species identified correctly?

Stage 3 checks this explicitly, because getting it wrong inflates conductivity
several-fold without any error being raised.

Both conductivity routes are quadratic in the charge vector:

```
Onsager:          sigma = e^2 * z^T L z
Nernst-Einstein:  sigma = e^2 * sum_i z_i^2 * D_i * N_i
```

so a species with `z = 0` contributes exactly zero — diagonal and cross terms
alike. EC, DMC and MA therefore cannot contribute to conductivity **provided the
charge vector lines up with the trajectory's per-species atom blocks**.

That alignment is the weak point. `onsager_calc` slices the trajectory using
`species_order`, but builds its charge vector from dict insertion order, and it
never checks that the two agree. If they ever diverge, a neutral solvent
silently picks up an ionic charge and nothing complains.

Stage 3 prints the layout it is actually using:

```
  species blocks in the trajectory:
         MA    atoms [    0: 1496]  z=+0  neutral
         EC    atoms [ 1496: 2516]  z=+0  neutral
         DMC   atoms [ 2516: 3740]  z=+0  neutral
         PF6   atoms [ 3740: 3978]  z=-1  IONIC
         LI    atoms [ 3978: 4012]  z=+1  IONIC
```

and then asserts:

1. the species order matches `system.top`'s `[ molecules ]` section — the order
   the atoms are physically laid out in;
2. the box is charge neutral;
3. at least one ionic species exists;
4. **a control run with the charges deliberately rolled onto the wrong species
   changes the answer.** If it did not, charges would not be reaching the
   conductivity sums at all.

For the shipped reference system that control is unambiguous:

| Charge assignment | sigma_onsager | sigma_NE |
|---|---|---|
| correct (PF6 −1, LI +1) | 19.08 mS/cm | 28.82 mS/cm |
| rolled onto neutral MA | 109.31 mS/cm | 111.77 mS/cm |

Nearly a 6x error, because MA contributes 136 fast neutral molecules against
only 34 PF6. Silent, and entirely plausible-looking if you were not checking.

## Troubleshooting

### `libOpenMMVelocityVerlet.so: cannot open shared object file`

By far the most common failure. The plugin's SWIG wrapper
(`_velocityverletplugin*.so`) is built **without RPATH/RUNPATH**, so it can only
find `libOpenMMVelocityVerlet.so` through `LD_LIBRARY_PATH`.

`submodules/openmm/install.sh` appends that export to `~/.bashrc` — but of the
user who ran the install. Since installing into `/usr/local/openmm` needs
`sudo`, that is usually **root's** `~/.bashrc`, not yours. And `~/.bashrc` is
only sourced by *interactive non-login* shells, so any job started with
`nohup`, `cron`, `systemd`, `bash -c`, an IDE, or a scheduler will not see it
either.

Permanent fix, independent of shell and user:

```bash
echo /usr/local/openmm/lib | sudo tee /etc/ld.so.conf.d/openmm.conf
sudo ldconfig
```

Per-shell fix:

```bash
export LD_LIBRARY_PATH=/usr/local/openmm/lib:$LD_LIBRARY_PATH
```

If the library does not exist at all, (re)run `submodules/openmm/install.sh`.

### `ModuleNotFoundError: No module named 'byteff2'`

ByteFF2 is not pip-installed; the examples put the repo root on `sys.path`
themselves. Run the script by path as shown above rather than copying it
elsewhere.

### Stage 3 values drift

Almost always a dependency version change. Compare your installed versions
against `requirements.txt` — particularly `torch`, `numpy`, `MDAnalysis` and
`scipy`. The comparison tolerance is 0.5% relative, far looser than any
physically meaningful difference, so a real drift means something changed in
the numerics.

## What is in `reference/`

Real output from a 1:10 LiPF6 in EC:DMC:MA = 3:3:4 run at 298 K (4012 atoms,
408 molecules), trimmed to keep the repository small:

| File | Contents |
|---|---|
| `nvt.dcd` | last **600 frames** (600 ps) of the 20 ns NVT trajectory, ~28 MB |
| `nvt_state.csv` | the matching 600 rows (used for box volume and temperature) |
| `viscosity.csv` | full 20000-row output of the 1 ns non-equilibrium shear run |
| `params/` | `system.top`, per-species `.itp`/`.atp`/`.gro`, and the force field parameter JSONs |

The trajectory is a **contiguous** slice, never strided: `onsager_calc` hardcodes
`dts_ps = range(50, 200)` and treats one frame as one picosecond (NVT runs at
2 fs with 500-step reporting). It also discards the first 200 frames and fits
the MSD out to 200 ps, so ~400 usable frames is the floor — 600 leaves margin.

> These numbers validate your installation; they are **not** converged transport
> properties. 600 ps is far too short for publication-quality diffusivities.
> Use a full protocol run for physics.

## Regenerating the expectations (maintainers)

```bash
python run_installation_test.py --regenerate-expected
```

Overwrites `expected_results.json` from the current machine. Only do this when
a change to the analysis is intended — it will happily bake in a regression.
