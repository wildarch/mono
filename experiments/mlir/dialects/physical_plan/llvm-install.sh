#!/bin/bash
set -e

cd ~/workspace/llvm-project-17.0.6.src/

mkdir -p build/
cd build/
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_USE_SANITIZER="Address;Undefined" \
   -DLLVM_USE_SPLIT_DWARF=ON \
   -DCMAKE_INSTALL_PREFIX=../../llvm-install