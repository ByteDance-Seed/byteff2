#!/bin/bash -l
######################################################################
## This script is used to compile libraries and to build Python 
## distribution for OpenMM with CUDA support.
##
## Build artifacts (.whl and .tar.gz files) and an installation script
## (install.sh) are stored in the `output` directory within 
## the top-level source directory.
##
## Explicitly set -D_GLIBCXX_USE_CXX11_ABI to ensure ABI 
## compatibility with pytorch.
## Check current CXX11_ABI setting of pytorch with:
## python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
##
## For compatibility with GCC 12, explicitly set the following options:
##   -DCMAKE_CXX_FLAGS_RELEASE:STRING="-O2 -DNDEBUG"
##   -DCMAKE_C_FLAGS_RELEASE:STRING="-O2 -DNDEBUG"
##
## Usage:
##   ./build.sh [PREFIX_DIR]
##   - [PREFIX_DIR] is optional. If not specified, defaults to:
##     /usr/local/openmm
######################################################################

## Exits at any error and prints each command
set -ex 

PROJECT="openmm"
echo "### Building ${PROJECT} ###"

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

## Checks that the top-level directory contains source code
TOP_LEVEL_DIR=$(pwd)
if ! [ -f "${TOP_LEVEL_DIR}/CMakeLists.txt" ]; then
  echo "Invalid top-level directory"
  exit 1
fi

## Creates an output directory to host build artifacts
OUTPUT_DIR="${TOP_LEVEL_DIR}/output"
rm -rf ${OUTPUT_DIR} && \
  mkdir -p ${OUTPUT_DIR}

## Sets the installation directory
PREFIX_DIR="${1:-/usr/local/${PROJECT}}"
echo "### PREFIX_DIR = ${PREFIX_DIR} ###"

## Creates and enters a clean `build` directory
rm -rf "${TOP_LEVEL_DIR}/build" && \
  mkdir -p "${TOP_LEVEL_DIR}/build" && \
  cd "${TOP_LEVEL_DIR}/build"

## Installs dependencies to compile libs
apt update && apt install -y cmake swig doxygen python3-venv

###########################################
### CPU & CUDA libs and Python wrappers ###
###########################################
cmake -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR} \
      -DCMAKE_CXX_FLAGS:STRING=-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI} \
      -DCMAKE_CXX_FLAGS_RELEASE:STRING="-O2 -DNDEBUG" \
      -DCMAKE_C_FLAGS_RELEASE:STRING="-O2 -DNDEBUG" \
      -DOPENMM_BUILD_AMOEBA_CUDA_LIB=ON \
      -DOPENMM_BUILD_AMOEBA_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_AMOEBA_PLUGIN=ON \
      -DOPENMM_BUILD_COMMON=OFF \
      -DOPENMM_BUILD_CPU_LIB=ON \
      -DOPENMM_BUILD_CUDA_LIB=ON \
      -DOPENMM_BUILD_CUDA_TESTS=OFF \
      -DOPENMM_BUILD_DRUDE_CUDA_LIB=ON \
      -DOPENMM_BUILD_DRUDE_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_DRUDE_PLUGIN=ON \
      -DOPENMM_BUILD_EXAMPLES=ON \
      -DOPENMM_BUILD_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_PME_PLUGIN=ON \
      -DOPENMM_BUILD_PYTHON_WRAPPERS=ON \
      -DOPENMM_BUILD_RPMD_CUDA_LIB=ON \
      -DOPENMM_BUILD_RPMD_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_RPMD_PLUGIN=ON \
      -DOPENMM_BUILD_SHARED_LIB=ON \
      -DOPENMM_BUILD_STATIC_LIB=OFF \
      -DOPENMM_GENERATE_API_DOCS=OFF \
      -DPYTHON_EXECUTABLE=$(which python3) ..

## Builds targets and fills ${PREFIX_DIR}
echo "### Compiling CPU and CUDA libs ###"
make -j$(nproc) install
echo "### Finished compiling CPU and CUDA libs ###"

echo "### Building Python distribution ###"
## Requires C++ libs
export OPENMM_INCLUDE_PATH=${PREFIX_DIR}/include
export OPENMM_LIB_PATH=${PREFIX_DIR}/lib

if [ "${GLIBCXX_ABI}" = "1" ]; then
  export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
else
  export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
fi
echo $CXXFLAGS

## Sets up a virtual environment for building
cat << EOF > ${TOP_LEVEL_DIR}/build/python/pyproject.toml
[build-system]
requires = ["setuptools", "wheel", "Cython>=3.0,<4", "oldest-supported-numpy"]
build-backend = "setuptools.build_meta"
EOF

cd ${TOP_LEVEL_DIR}/build/python
python3 -m pip install build && python3 -m build --wheel .
echo "### Finished building Python distribution ###"

## Exports build artifacts
## C++
tar -C "${PREFIX_DIR}/.." -czvf "${OUTPUT_DIR}/${PROJECT}.tar.gz" ${PROJECT}
## Python
cp ${TOP_LEVEL_DIR}/build/python/dist/*.whl ${OUTPUT_DIR}

## Prepares an installation script
cat << EOF > ${OUTPUT_DIR}/install.sh
#!/bin/bash -l
## Unzips compiled libraries to ${PREFIX_DIR}
## Installs Python wrappers
rm -rf ${PREFIX_DIR} && \\
  python3 -m pip uninstall -y ${PROJECT}
tar -C "${PREFIX_DIR%/*}" -xvf ${PROJECT}.tar.gz && \\
  python3 -m pip install --no-cache-dir --no-index *.whl

## Tests Python wrappers
# python3 -m ${PROJECT}.testInstallation

## Uninstalls compiled libs
# rm -rf ${PREFIX_DIR}

## Uninstalls Python wrappers
# python3 -m pip uninstall ${PROJECT}
EOF
chmod a+x ${OUTPUT_DIR}/install.sh

echo "### ${PROJECT} has been packaged in ${OUTPUT_DIR} ###"
