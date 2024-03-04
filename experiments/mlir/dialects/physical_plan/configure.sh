#!/bin/bash
rm -rf build/
cmake -S . -B build/ -G Ninja \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ROOT=~/workspace/llvm-install