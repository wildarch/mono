#!/bin/bash
rm -rf build/
cmake -S . -B build/ -G Ninja \
    -DParquet_DIR=/home/daan/workspace/apache-arrow-13.0.0/cpp/build/install/lib/cmake/Parquet \
    -DArrow_DIR=/home/daan/workspace/apache-arrow-13.0.0/cpp/build/install/lib/cmake/Arrow  \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCMAKE_BUILD_TYPE=Debug