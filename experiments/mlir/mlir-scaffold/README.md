# Generate scaffolding for an MLIR dialect
The idea is to create one place under experiments/ where all MLIR dialects live, and make an automated tool that can generate scaffolding for new dialects.
The resulting dialect should be a self-contained C++ library that can be included in other projects.

Files to generate:
- `CMakeLists.txt`
- `include/{path}/CMakeLists.txt` (calls `add_mlir_dialect`)
- `include/{path}/{dialect}Dialect.h`
- `include/{path}/{dialect}Dialect.td`
- `include/{path}/{dialect}Ops.h`
- `include/{path}/{dialect}Ops.td`
- `include/{path}/{dialect}Types.h`
- `include/{path}/{dialect}Types.td`
- `lib/{path}/CMakeLists.txt` (calls `add_mlir_dialect_library`)
- `lib/{path}/{dialect}Dialect.cpp`
- `lib/{path}/{dialect}Ops.cpp`
- `lib/{path}/{dialect}Types.cpp`
- `lib/{path}/opt/CMakeLists.txt`
- `lib/{path}/opt/main.cpp`