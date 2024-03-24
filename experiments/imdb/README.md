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

```bash
time cargo run --release --bin step2 ~/Downloads/imdb/title.principals.tsv.gz

> Final answer: nm0048389 (2989 characters)
> real    1m8,242s
```

Already a big improvement, almost 4x faster.

## 3. Bring out Perf
TODO: run perf to find bottlenecks.