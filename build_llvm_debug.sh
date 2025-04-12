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

# Configure, build and install
mkdir /tmp/build/

# Main differences with thirdpart/setup_llvm.sh from the avantgraph repo:
# - Build in Debug mode to enable debug utils such as the --debug flag
# - Install to /opt/llvm-debug
# - Does not build clang because we already install it using APT
cmake -G Ninja -S /opt/llvm-src/llvm -B /tmp/build \
   -DCMAKE_BUILD_TYPE=Debug \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-debug \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_LINK_LLVM_DYLIB=ON \
   -DLLVM_PARALLEL_LINK_JOBS=2 \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DLLVM_USE_LINKER=mold \

cmake --build /tmp/build
cmake --install /tmp/build

# Cleanup build dir so it is not included in the image snapshot
rm -r /tmp/build/
