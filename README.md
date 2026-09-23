<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)


# ByteFF2

<p align="center">
  <a href="https://arxiv.org/abs/2508.08575">
    <img src="https://img.shields.io/badge/ByteFF_Pol-arxiv-red"></a>
  <a href="http://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/License-Apache-blue"></a>
  <a href="https://huggingface.co/ByteDance-Seed/byteff2">
    <img src="https://img.shields.io/badge/🤗-HF%20Model-yellow"></a>
</p>

This repository provides two GNN-parameterized molecular mechanics force fields:

* [ByteFF](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc06640e) is a non-polarizable classical force field for drug-like molecules. Its edge-augmented, symmetry-preserving GNN predicts bonded and non-bonded parameters in a single pass and achieves state-of-the-art accuracy across torsional energy profiles, relaxed geometries, conformational energies, and off-equilibrium energies and forces.

* [ByteFF-Pol](https://www.nature.com/articles/s41467-026-73566-3) is a polarizable force field trained on high-level quantum mechanics (QM) data without experimental calibration. It accurately predicts thermodynamic and transport properties of small-molecule liquids and electrolytes, outperforming state-of-the-art traditional and machine learning force fields.

Released model weights are versioned separately from the force-field names:

* **ByteFF-26** is the 2026 version of the ByteFF model weights, trained on the [THEMol](https://github.com/ByteDance-Seed/THEMol) dataset using the joint-training workflow described in the [ByteFF paper](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc06640e).
* **ByteFF-Pol-25** is the 2025 release version of the ByteFF-Pol model weights.

## News
[2026/09/24]🔥ByteFF-26 is now available.  
[2025/08/25]🔥We release ByteFF-Pol-25.

## Getting started
### Prerequisites
* Python version >= 3.11

### Python Dependencies
All required Python packages are listed in `pyproject.toml`. To install them with uv, run:
```
uv sync
```
### Installing Gromacs
Download Gromacs from [official website](https://manual.gromacs.org/documentation/current/download.html).
```
wget https://ftp.gromacs.org/gromacs/gromacs-2025.3.tar.gz
```

To install Gromacs, please refer to the [official documentation](https://manual.gromacs.org/documentation/current/install-guide/index.html).
```
tar xfz gromacs-2025.3.tar.gz
cd gromacs-2025.3
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
make
make check
sudo make install
source /usr/local/gromacs/bin/GMXRC
```

### Installing OpenMM for ByteFF2

To run **ByteFF2**, you need a customized version of [**OpenMM**](https://github.com/openmm/openmm) and [**OpenMM-VelocityVerlet**](https://github.com/z-gong/openmm-velocityVerlet).

1. Navigate to the `submodules/openmm` directory:
   ```bash
   cd submodules/openmm
   ```

2. Run the installation script:
   ```bash
   ./install.sh [OPENMM_DIR]
   ```
   - `[OPENMM_DIR]` (optional): Installation path for OpenMM.
   - Default installation path is:
     ```
     /usr/local/openmm
     ```

3. The script will:
   - Compile and install the patched `openmm` (v8.3.1)
   - Compile and install `openmm-velocityVerlet`
   - Add required environment variables (`OPENMM_DIR` and `LD_LIBRARY_PATH`) to your `~/.bashrc`

4. After installation, restart your terminal or run:
   ```bash
   source ~/.bashrc
   ```

After successful installation, you should see:
```
Success: Installed OpenMM and openmm-velocityVerlet.
```

### Installing xtb

To install **xtb**, please refer to the [official documentation](https://xtb-docs.readthedocs.io/en/latest/setup.html).

Download a precompiled binary from the [latest release page](https://github.com/grimme-lab/xtb/releases/latest) and extract it:
```
wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
tar xf xtb-6.7.1-linux-x86_64.tar.xz
# The xtb executable is located in xtb-dist/bin after extraction.
export PATH=$PWD/xtb-dist/bin:$PATH
xtb --version
```

To set up the full xtb runtime environment, you can alternatively source the bundled environment script:
```
source $PWD/xtb-dist/share/xtb/config_env.bash
xtb --version
```

### Model Weights and Data
The model weights and example data are available on the
[`v1.1.0` branch of ByteDance-Seed/byteff2 on Hugging Face](https://huggingface.co/ByteDance-Seed/byteff2/tree/v1.1.0).
Use this branch with ByteFF2 1.1.0; the default Hugging Face branch may contain a different resource layout.

The repository contains:
- **ByteFF-26**:
  - `ByteFF-26/data/training/{hessian,torsion}/` — example training data.
  - `ByteFF-26/models/member-01` through `member-05` — the five-member ensemble; members use seeds 43 through 47.
- **ByteFF-Pol-25**:
  - `ByteFF-Pol-25/data/training/cluster/` — example training data and companion metadata.
  - `ByteFF-Pol-25/data/validation/` — validation dataset and its co-located configuration.
  - `ByteFF-Pol-25/model/` — the ByteFF-Pol model.

Install the Hugging Face CLI, choose a directory outside the Python package, and download the resources:

```bash
pip install -U "huggingface_hub[cli]"
export BYTEFF2_ASSET_ROOT="$HOME/.cache/byteff2/v1.1.0"
hf download ByteDance-Seed/byteff2 --revision v1.1.0 --local-dir "$BYTEFF2_ASSET_ROOT"
```

ByteFF2 reads resources directly from `BYTEFF2_ASSET_ROOT`; no files need to be copied into `byteff2/`.
Set this variable in each shell where you run the examples, or pass `asset_root` explicitly when loading a model.
Keep generated preprocessing and training outputs outside the asset directory. The `hf download` workflow does
not require Git LFS.

For model inference only, download the model you need instead of the full repository:

```bash
# ByteFF-26: all five ensemble members
hf download ByteDance-Seed/byteff2 --revision v1.1.0 \
  --include "ByteFF-26/models/*" --local-dir "$BYTEFF2_ASSET_ROOT"

# ByteFF-Pol-25
hf download ByteDance-Seed/byteff2 --revision v1.1.0 \
  --include "ByteFF-Pol-25/model/*" --local-dir "$BYTEFF2_ASSET_ROOT"
```

Training and validation examples also require their corresponding data:

```bash
# ByteFF training data
hf download ByteDance-Seed/byteff2 --revision v1.1.0 \
  --include "ByteFF-26/data/training/*" --local-dir "$BYTEFF2_ASSET_ROOT"

# ByteFF-Pol training and validation data
hf download ByteDance-Seed/byteff2 --revision v1.1.0 \
  --include "ByteFF-Pol-25/data/*" --local-dir "$BYTEFF2_ASSET_ROOT"
```

## Quick Start

### Load a released model

After configuring the asset root, load an official model by its release name:

```python
from byteff2.train import load_pretrained_model

byteff = load_pretrained_model("ByteFF-26")  # all five members, in evaluation mode
byteff_single = load_pretrained_model("ByteFF-26", member=1)
byteff_pol = load_pretrained_model("ByteFF-Pol-25", device="cpu")
```

Pass `asset_root="/path/to/byteff2-assets"` to override `BYTEFF2_ASSET_ROOT`. Model loading never downloads files
implicitly. ByteFF-26 requires all five members by default; missing members raise an error instead of silently
changing the ensemble. `member` is only supported for ByteFF-26. For custom models, the existing
`byteff2.train.load_model(model_directory)` interface remains available.

The old package-local asset paths and root-level `trained_models/` and `valid_data/` layouts are no longer supported.

You can refer to examples in the `example` directory; more details are available in the `README.md` file for each example.

### ByteFF examples
* `example/ByteFF/1_training` contains scripts for data preprocessing, GAFF2/AM1-BCC target generation, pretraining, and training ByteFF model.
* `example/ByteFF/2_write_params` contains scripts to generate force field parameters using a trained ByteFF model for MD simulations.

### ByteFF-Pol examples
* `example/ByteFF-Pol/0_data_preparation` contains scripts for preparing molecular data for training ByteFF-Pol.
* `example/ByteFF-Pol/1_training` contains scripts for training ByteFF-Pol.
* `example/ByteFF-Pol/2_compare_qm` contains scripts to compare QM and FF energies for dimers and clusters.
* `example/ByteFF-Pol/3_write_params` contains scripts to generate force field parameters using a trained ByteFF-Pol model.
* `example/ByteFF-Pol/4_MD_simulations` contains scripts for molecular dynamics (MD) simulations using ByteFF-Pol.
* `example/ByteFF-Pol/5_chemical_space` contains scripts for chemical-space coverage and similarity analysis using ByteFF-Pol.
* `example/ByteFF-Pol/6_cluster_pes` contains scripts to reproduce cluster potential energy surface (PES) validation using ByteFF-Pol.

## Run Tests
You can verify the environments by running the tests:
```
make test
```

## License
This project is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Citation
If you find ByteFF-Pol or ByteFF is useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex

@article{zheng2026bridging,
  title   = {Bridging quantum mechanics to liquid properties via a universal organic force field},
  author  = {Tianze Zheng and Xingyuan Xu and Zhi Wang and Zhenze Yang and Yuanheng Wang and Xu Han and Lei Chen and Zhenliang Mu and Ziqing Zhang and Siyuan Liu and Sheng Gong and Kuang Yu and Wen Yan},
  year    = {2026},
  journal = {Nature Communications},
  doi     = {10.1038/s41467-026-73566-3},
  url     = {https://www.nature.com/articles/s41467-026-73566-3}
}

@Article{D4SC06640E,
  author    = {Tianze Zheng and Ailun Wang and Xu Han and Yu Xia and Xingyuan Xu and Jiawei Zhan and Yu Liu and Yang Chen and Zhi Wang and Xiaojie Wu and Sheng Gong and Wen Yan},
  title     = {Data-driven parametrization of molecular mechanics force fields for expansive chemical space coverage},
  journal   = {Chem. Sci.},
  year      = {2025},
  pages     = {-},
  publisher = {The Royal Society of Chemistry},
  doi       = {10.1039/D4SC06640E},
  url       = {http://dx.doi.org/10.1039/D4SC06640E}
}

```

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
