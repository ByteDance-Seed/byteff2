######################################################################
## This script is used to compile libraries and to build Python 
## distribution for openmm-velocityVerlet.
##
## Usage:
##   ./build.sh [OPENMM_DIR]
##   - [OPENMM_DIR] is optional. If not specified, defaults to:
##     /usr/local/openmm
######################################################################

## Exits at any error and prints each command
set -ex 

# forbid using cmake 4
pip3 install cmake~=3.17

PROJECT="openmm_vv"
echo "### Building ${PROJECT} ###"

OPENMM_DIR="${1:-/usr/local/openmm}"

## Checks that the top-level directory contains source code
TOP_LEVEL_DIR=$(pwd)
if ! [ -f "${TOP_LEVEL_DIR}/CMakeLists.txt" ]; then
  echo "Invalid top-level directory"
  exit 1
fi

# Determine ABI flag
if python3 -c "import torch" >/dev/null 2>&1; then
    torch_version=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    glibcxx_abi=$(python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
    echo "Detected PyTorch version: $torch_version, CXX11_ABI=${glibcxx_abi}"
else
    glibcxx_abi="False"
    echo "PyTorch not found, defaulting CXX11_ABI=0"
fi
if [ "$glibcxx_abi" = "True" ]; then
    GLIBCXX_ABI=1
else
    GLIBCXX_ABI=0
fi
echo "Using -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI}"

## Creates an output directory to host build artifacts
OUTPUT_DIR="${TOP_LEVEL_DIR}/output"
mkdir -p ${OUTPUT_DIR}

## Sets the utility directory
CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

## Sets the installation directory
PREFIX_DIR="${OUTPUT_DIR}/${PROJECT}"
echo "### PREFIX_DIR = ${PREFIX_DIR} ###"

## Creates and enters a clean `build` directory
rm -rf "${TOP_LEVEL_DIR}/build" && \
  mkdir -p "${TOP_LEVEL_DIR}/build" && \
  cd "${TOP_LEVEL_DIR}/build"

###########################################
### CPU & CUDA libs and Python wrappers ###
###########################################
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR} \
      -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
      -DVELOCITYVERLET_BUILD_CUDA_LIB=ON \
      -DCMAKE_CXX_FLAGS:STRING=-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI} \
      -DOPENMM_DIR="${OPENMM_DIR}" \
      -DPYTHON_EXECUTABLE=$(which python3) \
      ${TOP_LEVEL_DIR}

## Builds C++ and CUDA targets and fills ${PREFIX_DIR}
echo "### Compiling CPU and CUDA libs ###"
make -j$(nproc) install
echo "### Finished compiling CPU and CUDA libs ###"

## Builds Python distribution
if [ "${GLIBCXX_ABI}" = "1" ]; then
  export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
else
  export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
fi

echo $CXXFLAGS

echo "### Building Python distribution ###"
make PythonInstall
cd ${TOP_LEVEL_DIR}/build/python
python3 -m pip install build && python3 -m build --wheel .
echo "### Finished building Python distribution ###"

## Exports build artifacts
## C++
tar -C "${OUTPUT_DIR}/${PROJECT}" -czvf "${OUTPUT_DIR}/${PROJECT}.tar.gz" .
rm -rf "${OUTPUT_DIR}/${PROJECT}"
## Python
cp ${TOP_LEVEL_DIR}/build/python/dist/*.whl ${OUTPUT_DIR}

## Prepares an installation script
cat << EOF > ${OUTPUT_DIR}/install.sh
#!/bin/bash -l
## Unzips compiled libraries to /usr/local/openmm
## Installs Python wrappers
OPENMM_DIR="${OPENMM_DIR}"
tar -C \${OPENMM_DIR} -xvf ${PROJECT}.tar.gz && \\
  python3 -m pip install --no-cache-dir --no-index --force-reinstall *.whl

## Uninstalls compiled libs
# rm -rf \${OPENMM_DIR}/include/openmm/VVIntegrator.h
# rm -rf \${OPENMM_DIR}/include/openmm/VVKernels.h
# rm -rf \${OPENMM_DIR}/lib/libOpenMMVelocityVerlet.so
# rm -rf \${OPENMM_DIR}/lib/plugins/libVelocityVerletPluginCUDA.so

## Uninstalls Python wrappers
# python3 -m pip uninstall velocityverletplugin
EOF
chmod a+x ${OUTPUT_DIR}/install.sh

echo "### ${PROJECT} has been packaged in ${OUTPUT_DIR} ###"