# Watcher
Automatically runs build/test after you modify source files.

Bazel query can help implement this.

To get locations for all files a target depends on:
```
bazel query 'deps(//experiments/superg:superg)' --output location
```

This will give an output of the form:
```
/home/daan/workspace/mono/experiments/superg/build.rs:1:1: source file //experiments/superg:build.rs
/home/daan/workspace/mono/experiments/superg/BUILD.bazel:16:19: cargo_build_script rule //experiments/superg:build_script
/home/daan/workspace/mono/experiments/superg/BUILD.bazel:16:19: rust_binary rule //experiments/superg:build_script_
/home/daan/workspace/mono/experiments/superg/src/ast.rs:1:1: source file //experiments/superg:src/ast.rs
/home/daan/workspace/mono/experiments/superg/src/bracket.rs:1:1: source file //experiments/superg:src/bracket.rs
...
```

It also shows files from the cache, which we should throw out: we only care about files that live under the repository root.

We can use golang's [`fsnotify`](https://github.com/fsnotify/fsnotify).

So:
1. Run the command, capture the output as a string.
2. Filter out paths outside the project root.
3. Trim the line numbers and notes to the right of the first `:`.
4. Trim the filename, we only care about the directory it is in.
4. Deduplicate the list.
5. Watch all of those paths for changes, and rebuild the target if anything changes. 

`BUILD` files and `WORKSPACE` are special: if any of those files change, we should re-run the query command.
