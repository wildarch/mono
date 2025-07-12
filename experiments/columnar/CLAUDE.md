# Building
- `cmake --build experiments/columnar/build` builds all targets
- When changing rewrite rules or passes, only build the `columnar-opt` target.
- When changing SQL parsing logic, build `translate`
- When changing the runtime (used to execute the lowered queries), build `execute`

# Testing
- Prefer to run tests on individual files.
  The `// RUN:` comments in test files show the commands to execute.
- To run the full testsuite, run `cmake --build experiments/columnar/build --target check`.
  Do not run this unless explicitly asked.

# MLIR Source Code
If you need to use an MLIR API for which there are no examples in the code base, check the MLIR source code on how to use it.
Available from @/opt/llvm-src/mlir.
