#!/bin/bash
set -e

PROJECT_ROOT=experiments/copy-and-patch/

cd $PROJECT_ROOT

rm -rf build/
mkdir build/
cd build/

# Initial example
clang-20 -O3 -mcmodel=medium -c ../stencils.c
objdump -d -Mintel,x86-64 --disassemble --reloc stencils.o

clang-20 ../cnp_jit.c ../cnp_stencils.c -o cnp_jit

# Complex stencil
clang-20 -O3 -mcmodel=medium -c ../complex_stencil.c
objdump -d -Mintel,x86-64 --disassemble --reloc complex_stencil.o
