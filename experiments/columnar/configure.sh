#!/bin/bash
WORKSPACE_ROOT=experiments/columnar
BUILD_DIR=$WORKSPACE_ROOT/build
rm -rf $BUILD_DIR
cmake -S $WORKSPACE_ROOT -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER=clang++-20  \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_LINKER_TYPE=MOLD \
    -DLLVM_ROOT="/opt/llvm-debug" \
