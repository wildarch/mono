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
