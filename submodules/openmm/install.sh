######################################################################
## This script is used to compile and install openmm 
## and openmm-velocityVerlet.
##
## Usage:
##   ./install.sh [OPENMM_DIR]
##   - [OPENMM_DIR] is optional. If not specified, defaults to:
##     /usr/local/openmm
######################################################################

# Exits at any error and prints each command
set -ex

OPENMM_DIR="${1:-/usr/local/openmm}"
if [[ "$OPENMM_DIR" != /*/openmm ]]; then
    echo "Error: OPENMM_DIR must be an absolute path ending with '/openmm'"
    exit 1
fi
if [ -e "$OPENMM_DIR" ]; then
    echo "Error: Directory '$OPENMM_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi
if [ ! -w "$(dirname "$OPENMM_DIR")" ]; then
    echo "Error: No write permission to $(dirname "$OPENMM_DIR")"
    echo "Please run with sudo or choose a directory you have write access to."
    exit 1
fi

TOP_LEVEL_DIR=$(pwd)

rm -rf "${TOP_LEVEL_DIR}/openmm"
rm -rf "${TOP_LEVEL_DIR}/openmm-velocityVerlet"

# Install openmm
git clone --branch 8.3.1 --single-branch https://github.com/openmm/openmm.git
cd openmm
git am ${TOP_LEVEL_DIR}/amoeba_scale_cpu.patch
git am ${TOP_LEVEL_DIR}/amoeba_scale_cuda.patch
cp "${TOP_LEVEL_DIR}/build_openmm.sh" ./
./build_openmm.sh ${OPENMM_DIR}
cd output
./install.sh
cd ${TOP_LEVEL_DIR}
# test openmm installation
# python3 -m openmm.testInstallation

# Install openmm-velocityVerlet
git clone https://github.com/z-gong/openmm-velocityVerlet.git
cd openmm-velocityVerlet
cp "${TOP_LEVEL_DIR}/build_openmm_vv.sh" ./
./build_openmm_vv.sh ${OPENMM_DIR}
cd output
./install.sh
cd ${TOP_LEVEL_DIR}

# Environment variables
if ! grep -q "export OPENMM_DIR=" ~/.bashrc; then
    {
        echo "export OPENMM_DIR=\"$OPENMM_DIR\""
        echo "export LD_LIBRARY_PATH=\"\$OPENMM_DIR/lib:\$LD_LIBRARY_PATH\""
    } >> ~/.bashrc
    echo "OPENMM_DIR and LD_LIBRARY_PATH have been added to ~/.bashrc"
else
    echo "OPENMM_DIR already exists in ~/.bashrc, skip."
fi

echo "Success: Installed OpenMM and openmm-velocityVerlet."