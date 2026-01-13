---
license: apache-2.0
tags:
- chemistry
- biology
---
# ByteFF2

This repository contains the model used for the paper [Bridging Quantum Mechanics to Organic Liquid Properties via a Universal Force Field](https://arxiv.org/abs/2508.08575)。

[ByteFF-Pol](https://arxiv.org/abs/2508.08575) is a polarizable force field parameterized by a graph neural network (GNN), trained on high-level quantum mechanics (QM) data, thus eliminating the need for experimental calibration. ByteFF-Pol achieves exceptional accuracy in predicting the thermodynamic and transport properties of small-molecule liquids and electrolytes, outperforming SOTA traditional and ML force fields

# Trained Models
The `trained_models` folder contains the trained model for ByteFF-Pol and its corresponding configuration (.yaml) file.

# How to use
Code and examples are available in the [byteff2](https://github.com/ByteDance-Seed/byteff2) repository.

## MD Protocol Configuration
The MD protocols (density, transport, hvap) support configuring timesteps and durations via the JSON config passed to the runners.

- Time control (alternatives):
  - Provide explicit steps: `npt_steps`, `nvt_steps`, `nonequ_steps`.
  - Or provide duration: `npt_time_ns`/`npt_time_ps`, `nvt_time_ns`/`nvt_time_ps`, `nonequ_time_ns`/`nonequ_time_ps`.
    Steps are computed from time using the selected timestep.

- Timestep (fs):
  - `npt_timestep_fs` (default 2)
  - `nvt_timestep_fs` (default 2)
  - `nonequ_timestep_fs` (default 1; VVIntegrator)

Example snippet:

```
{
  "protocol": "Transport",
  "temperature": 298,
  "npt_time_ns": 2.0,
  "nvt_time_ns": 10.0,
  "nonequ_time_ns": 1.0,
  "npt_timestep_fs": 1,
  "nvt_timestep_fs": 1,
  "nonequ_timestep_fs": 1
}
```

Notes:
- Smaller timesteps improve stability at the cost of runtime.
- If both steps and time are provided, steps take precedence.

### Stage file overrides (handoffs)
- `npt_state_csv`, `npt_dcd`: explicit paths for NPT outputs. Used when rescaling NPT→NVT and during safe-resume on NPT.
- `nvt_state_csv`, `nvt_dcd`: explicit paths for NVT outputs. Used when seeding nonequilibrium and during safe-resume on NVT.
- If not set, files are read from `output_dir` by default.

### Resume behavior and caching
- When `resume: true` and previously generated artifacts exist under `params_dir`, the runners:
  - Reuse cached force-field files (`.itp/.atp/.gro`) instead of regenerating them.
  - Reuse the packed system (`system.top` and `solvent_salt.gro`) and skip box packing, starting MD from the selected stage (NPT/NVT/nonequ).
- To force regeneration, set `"force_regenerate_params": true` or delete cached files in `params_dir`.

### Safe resume (NaN fallback)
If a resume attempts to continue from a checkpoint and OpenMM reports a NaN (commonly due to a partially written checkpoint), the runner will automatically fall back to a stable state derived from the last usable trajectory frame.

Behavior:
- Load the last stable DCD frame (skipping the last N frames via backoff) and set positions from it.
- Set a diagonal periodic box using the corresponding state CSV row (or the last row if shorter).
- Reassign velocities to the target temperature and optionally minimize briefly.
- Warm-up (optional): temporarily disable the barostat and reduce the integrator step size for a short period to stabilize, then restore settings and continue to the original target step count.

Config keys:
- `resume_safe_backoff_frames` (int, default 2): how many final frames to skip when picking the fallback frame from DCD.
- `resume_safe_minimize` (bool, default true): run a brief minimization after applying fallback positions.
- `resume_safe_warmup_steps` (int, default 5000): warm-up steps to run before continuing.
- `resume_safe_warmup_step_factor` (float, default 2.0): warm-up uses `base_step / factor`.
- `resume_safe_disable_barostat_warmup` (bool, default true): temporarily disable MonteCarloBarostat during warm-up.

Example snippet:

```
{
  "resume": true,
  "npt_state_csv": "./npt_state.csv",
  "npt_dcd": "./npt.dcd",
  "nvt_state_csv": "./nvt_state.csv",
  "nvt_dcd": "./nvt.dcd",
  "resume_safe_backoff_frames": 4,
  "resume_safe_minimize": true,
  "resume_safe_warmup_steps": 20000,
  "resume_safe_warmup_step_factor": 2.0,
  "resume_safe_disable_barostat_warmup": true
}
```

## Citation
If you find ByteFF-Pol is useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex

@misc{zheng2025bridgingquantummechanicsorganic,
  title         = {Bridging Quantum Mechanics to Organic Liquid Properties via a Universal Force Field},
  author        = {Tianze Zheng and Xingyuan Xu and Zhi Wang and Xu Han and Zhenliang Mu and Ziqing Zhang and Sheng Gong and Kuang Yu and Wen Yan},
  year          = {2025},
  eprint        = {2508.08575},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  url           = {https://arxiv.org/abs/2508.08575}
}
```
