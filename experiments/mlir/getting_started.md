# Getting started with MLIR
This documents my getting started with MLIR.

We start at https://mlir.llvm.org/getting_started/.

My clone of `llvm-project` has a commit hash `2aea0a9de093624b39cf919af8d2755fe9cfec5a`` as HEAD.
Following the advice, I change the configuration command to:

```shell
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DCMAKE_INSTALL_PREFIX=/home/daan/workspace/llvm-install \
   -DLLVM_INSTALL_UTILS=ON
```

This also includes setting up an installation prefix and enabling utils, which the standalone build (see below) needs.
We omit the sanitizers because they cause linker errors in the standalone project.

I do not have enough RAM for the default number of threads running in parallel, so I build LLVM using: 

```shell
ninja -j 6
```

We build the full project rather than just `check-mlir` because the installer requires it.
Without it I get this error:

```
CMake Error at lib/FuzzMutate/cmake_install.cmake:46 (file):
  file INSTALL cannot find
  "/home/daan/workspace/llvm-project/build/lib/libLLVMFuzzerCLI.a": No such
  file or directory.
Call Stack (most recent call first):
  lib/cmake_install.cmake:48 (include)
  cmake_install.cmake:70 (include)
```

Install LLVM:

```shell
cmake --install .
```

Build the toy example:

```shell
ninja toyc-ch2
./bin/toyc-ch2 ../mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo
```

Now we move on to the standalone dialect example.
Copy it to a new directory:
```
cp -r ../mlir/examples/standalone ~/workspace/mlir-standalone
cd ~/workspace/mlir-standalone
```

We follow the README there:

```shell
mkdir build && cd build
cmake -G Ninja .. \
   -DMLIR_DIR=/home/daan/workspace/llvm-install/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=/home/daan/workspace/llvm-project/build/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

cmake --build . --target check-standalone
```

We should see passing tests:

```
Testing Time: 0.04s
  Unsupported: 1
  Passed     : 4
```