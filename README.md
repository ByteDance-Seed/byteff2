# ByteFF2

<p align="center">
  <a href="https://arxiv.org/abs/2508.08575">
    <img src="https://img.shields.io/badge/ByteFF_Pol-arxiv-red"></a>
  <a href="http://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/License-Apache-blue"></a>
  <a href="https://huggingface.co/ByteDance-Seed/byteff2">
    <img src="https://img.shields.io/badge/🤗-HF%20Model-yellow"></a>
</p>

This is the source repository for ByteFF-Pol.

* [ByteFF-Pol](https://arxiv.org/abs/2508.08575) is a polarizable force field parameterized by a graph neural network (GNN), trained on high-level quantum mechanics (QM) data, thus eliminating the need for experimental calibration. ByteFF-Pol achieves exceptional accuracy in predicting the thermodynamic and transport properties of small-molecule liquids and electrolytes, outperforming SOTA traditional and ML force fields.

## News
[2025/08/25]🔥We release ByteFF-Pol.


## Getting started
### Prerequisites
* Python version >= 3.11

### Python Dependencies
All required Python packages are listed in requirements.txt. To install them, run:
```
pip install -r requirements.txt
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
**My note:** I was having a lot of issues with the installation of their custom `openmm-velocityVerlet` library across MacOS and linux on two different HPCs with different machine configurations and environments. While I finally got `openmm-velocityVerlet` to install using pip, it would throw a segmentation error with the installation. Please note that `openmm-velocityVerlet` is only used for non-equilibrium MD run for viscosity calculations, so if you are interested in getting viscosity of electrolytes, this installation can be skipped. I have made several changes in this codebase to improve functionality and make it as generic as possible.

### Trained Models
The model and configuration file are available on HuggingFace [byteff2](https://huggingface.co/ByteDance-Seed/byteff2).

To download the model and configuration file, run:
```
pip install -U "huggingface_hub[cli]"
hf download ByteDance-Seed/byteff2 --local-dir byteff2
```

## Quick Start
You can refer to several examples in the · directory; more details are available in the README.md file for each example.

* `example/1_training` contains scripts for training ByteFF-Pol.
* `example/2_compare_qm` contains scripts to compare QM and FF energies.
* `example/3_write_params` contains scripts to generate force field parameters using trained ByteFF-Pol model.
* `example/4_MD_simulations` contains scripts for molecular dynamics (MD) simulations using ByteFF-Pol.
* `example/5_similarity` contains scripts for similarity analysis using ByteFF-Pol.

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

@misc{zheng2025bridgingquantummechanicsorganic,
  title         = {Bridging Quantum Mechanics to Organic Liquid Properties via a Universal Force Field},
  author        = {Tianze Zheng and Xingyuan Xu and Zhi Wang and Xu Han and Zhenliang Mu and Ziqing Zhang and Sheng Gong and Kuang Yu and Wen Yan},
  year          = {2025},
  eprint        = {2508.08575},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  url           = {https://arxiv.org/abs/2508.08575}
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