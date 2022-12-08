# Superg runtime
Superg is an evaluator for lazy functional languages based Combinatory Logic. 
It provides:
* Translation from an enriched lambda calculus into combinators using Kiselyov's semantic translation.
* A state-of-the-art combinator graph reducer (WIP).
* A simple LISP-like frontend

I hope to expand this README at some point. 
For now please refer to the report under `doc/paper.tex`. 

## Building
To build this crate, [install a Rust toolchain](https://rustup.rs) and run:

```shell
cargo build
```

Alternatively, you can [install Bazel](https://bazel.build/install) and run:

```shell
bazel build //experiments/superg
```