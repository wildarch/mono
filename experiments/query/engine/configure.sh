#!/bin/bash
rm -rf build/
cmake -S . -B build/ -G Ninja \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DParquet_DIR=/home/daan/workspace/apache-arrow-13.0.0/cpp/build/install/lib/cmake/Parquet \
    -DArrow_DIR=/home/daan/workspace/apache-arrow-13.0.0/cpp/build/install/lib/cmake/Arrow  \
    -DCMAKE_BUILD_TYPE=Debug