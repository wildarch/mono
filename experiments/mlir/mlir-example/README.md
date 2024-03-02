# MLIR Experiments
A tiny MLIR dialect.
This code does *not* build using Bazel (although it could be made to) since I need to learn how to build this with CMake.

## Building
```shell
mkdir build && cd build
cmake -G Ninja .. \
   -DMLIR_DIR=/home/daan/workspace/llvm-install/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=/home/daan/workspace/llvm-project/build/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=1

cmake --build . --target check-standalone
```