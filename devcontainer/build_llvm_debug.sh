#!/bin/bash
set -e

LLVM_VERSION=20.1.2
LLVM_PROJECT_FILE_NAME=llvm-project-${LLVM_VERSION}.src
LLVM_PROJECT_SRC_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVM_PROJECT_FILE_NAME}.tar.xz"

# Download and extract sources
wget $LLVM_PROJECT_SRC_URL -O /tmp/$LLVM_PROJECT_FILE_NAME.tar.xz
sha256sum -c - <<EOF
f0a4a240aabc9b056142d14d5478bb6d962aeac549cbd75b809f5499240a8b38  /tmp/llvm-project-20.1.2.src.tar.xz
EOF

# Extract to /opt/llvm-src
mkdir -p /opt/llvm-src
tar -xf /tmp/$LLVM_PROJECT_FILE_NAME.tar.xz \
   -C /opt/llvm-src --strip-components 1
rm /tmp/$LLVM_PROJECT_FILE_NAME.tar.xz

# Configure, build and install
mkdir /tmp/build/

# Based on https://mlir.llvm.org/getting_started/ with some tweaks:
# - Build in Debug mode to enable debug utils such as the --debug flag
# - Use clang-20 and mold
# - Install to /opt/llvm-debug
# - Reduced number of parallel link jobs to avoid running out of memory
cmake -G Ninja -S /opt/llvm-src/llvm -B /tmp/build \
   -DCMAKE_BUILD_TYPE=Debug \
   -DCMAKE_C_COMPILER=clang-20  \
   -DCMAKE_CXX_COMPILER=clang++-20 \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-debug \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_LINK_LLVM_DYLIB=ON \
   -DLLVM_PARALLEL_LINK_JOBS=2 \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DLLVM_USE_LINKER=mold \
   -DLLVM_USE_SPLIT_DWARF=ON \

cmake --build /tmp/build
cmake --install /tmp/build

# Cleanup build dir so it is not included in the image snapshot
rm -r /tmp/build/
