# Parsing IMDb at Break-Neck Speed
IMDb kindly provides its datasets free to use for non-commercial projects, see [here](https://developer.imdb.com/non-commercial-datasets/).
We will use it for a little case study on quickly parsing a common data format: GZIP'ed CSV files.

The biggest file is `title.principals.tsv.gz` at 626M.
How many lines is that?

```bash
gzip --stdout --decompress ~/Downloads/imdb/title.principals.tsv.gz | wc -l

> 84586851
```

I formulate an example query for this dataset that requires parsing and reading the entire file:

> Find the unique identifier of the person with the greatest number of distinct characters played.

## 1. Python
Python is my go-to tool when I need some data wrangling and don't care too much about performance, let us start there.


```bash
time ./experiments/imdb/step1.py ~/Downloads/imdb/title.principals.tsv.gz

> Final answer: nm0048389 (2994 characters)
> real    4m13,713s
```

## 2. Rust
A reasonably idiomatic Rust implementation.
* `flate2` library for decoding gzip
* `serde` for tsv deserialization. 
  We also borrow list decoding from `serde_json`.

```bash
time cargo run --release --bin step2 ~/Downloads/imdb/title.principals.tsv.gz

> Final answer: nm0048389 (2989 characters)
> real    1m8,242s
```

Already a big improvement, almost 4x faster.

## 3. Bring out Perf
We need some additional setup to enable effective profiling with perf.
Following [The Rust Performance Book](https://nnethercote.github.io/perf-book/profiling.html), we enable debug symbols in release binaries by adding to `Cargo.toml`:

```toml
[profile.release]
debug = 1
```

Enable running perf without root (global but temporary setting):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
```

Now we can record some perf data:

```bash
cargo build --release
perf record --call-graph dwarf target/release/step2 ~/Downloads/imdb/title.principals.tsv.gz
```

This writes perf logs to a file `perf.data`.

I like to use the [Hotspot](https://github.com/KDAB/hotspot) GUI for viewing perf logs:

```bash
hotspot ./perf.data
```
