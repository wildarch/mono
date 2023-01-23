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

## Analyzing TIGRE performance
### Cachegrind
Build the `tigre_fib` binary, and profile it using `cachegrind`:

```shell
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --bin tigre_fib
valgrind --tool=cachegrind ../../target/release/tigre_fib
```

Analyze the result with `kcachegrind cachegrind.out.*`.

### Callgrind
Build the `tigre_fib` binary, and profile it using `callgrind`:

```shell
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --bin tigre_fib
valgrind --tool=callgrind ../../target/release/tigre_fib
```

Analyze the result with `kcachegrind callgrind.out.*`.

### Flamegraph
Install the necessary dependencies:

```shell
cargo install flamegraph
sudo apt install -y linux-perf
```

Enable profiling:

```shell
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
export CARGO_PROFILE_RELEASE_DEBUG=true
```

Generate the flamegraph:

```shell
cargo flamegraph --bin tigre_fib
```

Now open `flamegraph.svg` in a browser.