# Columnar IR for Databases
Motivation: Analytical databases today are column-oriented, both in the way data is stored (split per column) and how it is processed (vectorized).

Problem: The internal representation of queries remains very close to relational algebra, modelling operators that transform tuples. This does not match well with how queries are actually executed, which leads to two problems:
1. Lowering needs to map the _tuple-wise_ operators to _columnar_ operations.
   Converting between these two models makes that lowering more complex.
2. We are tempted to make optimization decisions that make sense when processing per-tuple, but have detrimental effects in a per-column setting.

Solution: We present an intermediate query representation design that is equivalent and indeed very similar to relational algebra, but operates over columns instead of tables.
This makes lowering to a vectorized execution plan simpler without complicating high-level optimizations. It also addresses the problem of how to refer to identify column as described [here](https://xuanwo.io/2024/02-what-i-talk-about-when-i-talk-about-query-optimizer-part-1/).

## Operators
We consider the relational algebra consisting of the following operators:
- Access reads a base table from storage
- Aggregate groups by 0 or more key columns, and aggregates the remaining columns using an aggregator such as `SUM`, `MIN` or `COUNT`.
- Constant represents a constant table. Can also be seen as a special case of Access.
- Join produces the cartesian product of the tuples in two input relations.
- Projection reorders, removes or adds columns from input tuples. It can also contain per-tuple arithmetic expressions. For example, a projection may add the values of two input columns `a` and `b` together to produce a new output column `c`.
- Selection applies a filter to tuples in the input, removing all tuples that do not satisfy the predicate.
- Union concatenates the tuples of two relations into one output relation. The input relations must have the exact same set of columns.

They can be expressed with columnar operators as follows:
- Access: Multiple independent `read_column` ops
- Aggregate: `aggregate` takes in a list of group-by columns and a list of columns to aggregate.
  It also stored the aggregator per aggregated column.
  Returns columns represented the post-aggregation group-by keys and aggregated values.
- Constant: `constant` ops for individual columns.
- Join: A list of columns for the left side, and another list of columns for the right side.
  Returns new columns for all inputs.
- Projection: reorders and removals are no longer necessary and can be omitted entirely.
  Arithmetic is done with dedicated ops such as `add` to produce a new column based on two input columns.
- Selection: `select` takes in a list of columns to filter, as well as a list of columns needed to compute that filter (these may be the same as columns to filter).
  The filter to apply is embedded inside the op, and is composed of arithmetic expressions over columns.
  Multiple filter conditions can be specified, which are applied conjunctively.
  Returns new filtered columns for all columns to filter.
- Union: Takes in two lists of columns, which must be of pair-wise equal type.
  Return new columns that represent the concatenation of the inputs.

## Query Optimization
Transformations to consider:
- Constant propagation: Becomes simpler because we always propagate per-column constant information.
- Push down predicates
- Join order optimization: Done exactly the same way as in regular relational algebra.
- Pull up projections: Easy to pull up individual columns
- Common sub expression elimination: Is largely the same as in the per-tuple case.

### Predicate Pushdown
Given the child operator, pushdown is possible under the following conditions:
- Access: Never
- Aggregate: All columns are in the group-by clause.
- Constant: Directly apply the filter to the constant. Note: Not part of a classical pushdown.
- Join: The filter exclusively uses columns from a single side of the join.
  Otherwise, it is a join condition and belongs directly after the join.
- Projection: Always possible, by including the projection body in the predicate.
  May be undesirable if the computation is expensive, but it seems possible to fix that with CSE later.
- Selection: Can be merged into a combined `SelectOp` with both predicates.
- Union: Duplicate and apply to all children.

The predicates are stored inside the `SelectOp`, so they are moved with the op itself automatically.

### Projection Pull-up
In our IR, projection pull-up corresponds to arithmetic over columns.
We can delay such an operation `A` if the output appears as an input to:
- Access: N/A, has no children.
- Aggregate: If `A` is used by an aggregator it cannot be pulled up further.
  If the output of `A` is a group-by key, then we can instead put its input columns into the group-by key, and place `A` after the aggregation.
  This is generally if it does not make the group-by key much larger, i.e. `A` should not have too many input columns.
- Constant: N/A, has no children.
- Join: Useful if the output of `A` is not used in the join predicate.
  If it is part of the predicate, we can still do the pull-up, but it will offer little benefit, similar to the case with Aggregate.
- Selection: This is equivalent to predicate pushdown.
- Union: Only possible if all children include the same operation, and has not perf. benefit.
  We ignore it.

### Join Order Optimization
We start by looking for equi-join possibilities: Cases where the join can be expressed using a set of equality predicates between columns on the left-hand side and the right-hand side.
This will cover almost all common joins (the remaining cases can fall back to a simple cartesian product).
In some cases a bit of arithmetic (basically preprocessing one of the join sides) is needed before we can compare the columns: this computation needs to be moved outside the join tree first.

We can then extract the join tree and run any standard optimizer algorithm on it.

### Common Sub Expression Elimination
Common sub expressions may occur between arithmetic to produce columns and selection predicates.
This can be resolved once selection vectors are made explicit.

A standard _available expressions_ dataflow analysis suffices in most cases.
This could be extended with detection of common operations across union children.

## TPC-H Coverage
To support all TPC-H queries, we would need the following 'advanced' features:
- `ORDER BY`: Sorting the final output
- `LIMIT`: Top-K operator
- Correlated sub-queries
- `EXISTS`
- `NOT EXISTS`
- `SUM` inside `SUM`
- `OUTER JOIN`

DuckDB makes generating the TPC-H dataset easy:

```sql
CALL dbgen(sf = 1);

EXPORT DATABASE 'tpch-sf1' (FORMAT PARQUET);
```

## Lowering
### Query Plan to Pipelines
Ops to split:
- `AggregateOp`
- `JoinOp`
- `OrderByOp`
- `LimitOp`

Special case: `UnionOp`

Procedure:
1. Split up ops so that no op is both source and a sink at the same time.
   *NOTE:* requires that we establish blocking relationships between operators. e.g. aggregate output must wait until we complete the build.
2. Starting from sinks, find all ops that transitively feed column data into that sink.
   These ops will constitute the pipeline.
   Because sinks never output column data, we will only add source and transform ops.
   The sink op is *moved* into the pipeline (cloning it could have side effects).
   The source and transform ops are *cloned* because they do not have side effects.
   We only need to make sure that we also clone blocking relationships of sources.

## Pipeline
- translate (`--import-sql`)
- TODO: Constant propagation
- `push-down-predicates`
- TODO: Join order optimization
- TODO: Pull up projections
- TODO: Common sub-expression elimination
- `add-selection-vectors`
- TODO: Make NULLs explicit
- `group-table-reads`
- `make-pipelines`
- `lower-pipelines`
- `one-shot-bufferize` + `linalg-to-loops`
- `lower-to-llvm`
- execute

```bash
cmake --build experiments/columnar/build

experiments/columnar/build/translate --data /host-home-folder/Downloads/tpch-sf1 --import-sql experiments/columnar/test/sql/read.sql > /tmp/bug.mlir

experiments/columnar/build/columnar-opt /tmp/bug.mlir \
    --push-down-predicates \
    --add-selection-vectors \
    --make-pipelines \
    --group-table-reads \
    --lower-pipelines \
    --one-shot-bufferize --convert-linalg-to-loops \
    --lower-to-llvm \
    > /tmp/exec.mlir

experiments/columnar/build/execute /tmp/exec.mlir
```

## Runtime
### Interaction with Parquet readers.
There are a handful of physical column types
- BOOLEAN: 1 bit boolean
- INT32: 32 bit signed ints
- INT64: 64 bit signed ints
- INT96: 96 bit signed ints
- FLOAT: IEEE 32-bit floating point values
- DOUBLE: IEEE 64-bit floating point values
- BYTE_ARRAY: arbitrarily long byte arrays
- FIXED_LEN_BYTE_ARRAY: fixed length byte arrays

These require dedicated endpoints in the runtime.

A claimed range now consists of:
- The row group index (`int`)
- Offset (number of values to skip within the row group `std::int64_t`)
- The size of the chunk

## Hash Joins
The build has 4 phases:
1. Tuple collection: call `addTuple` for all tuples in the input.
   Does not require synchronization between threads.
2. Merge partitions (**Global, main thread**): Move tuple buffers between
   threads so that all data for one partition is assigned to a single thread.
3. Allocate the final tuple storage: Dependending on the number of tuples
   collected in the previous step, we allocated memory for the directory and the
   tuple storage.
4. Post processing: write the data for all partitions out to the final storage.

(2) is the most interesting bit here.
- We use a chained-style design rather than open
  addressing: The hash table contains pointers to the list of tuples per hash.
- Since the upper 16 bits of 64-bit pointers are unused, we can use them to store a bloom filter.
- The hash function can be constructed from CRC hardware instructions for best
  performance.
- To find the sizes of the per-hash lists, we can keep a counter during the
  initial buffering phase.
- Instead of a traditional chained table, we adopt the *unchained* hash table
  design. Tuples are stored in a contiguous array ordered by their hash prefix.
  The directory records the start of the range per hash prefix.

We need the following specialized low-level MLIR ops:
- CRC32 `(key: i32, seed: i32) -> i32`
- `VecPushOp` and others, equivalent to `std::vector`
- `CountLeadingZeroesOp`: Already in `LLVM` and `math` dialects!

The CRC32 op can be created in LLVM IR as:

```mlir
llvm.func @crc32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = llvm.call_intrinsic "llvm.x86.sse42.crc32.32.32"(%arg0, %arg1) : (i32, i32) -> i32
  llvm.return %0 : i32
}
```

MLIR Ops:
```mlir
#table_region = #columnar.table<"region" path="experiments/columnar/test/sql/data/region.parquet">
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<i64, si32, !columnar.str>
!ht = !columnar.hash_table<(si32) -> (!columnar.str)>
columnar.query {
  %0 = columnar.read_column #column_region_r_regionkey : <si32>
  %1 = columnar.read_column #column_region_r_name : <!columnar.str>

  %2 = columnar.global !buf
  columnar.hj.collect
    keys=[%0] : !columnar.col<si32>
    values=[%1] : !columnar.col<!columnar.str>
    -> %2 : !buf

  %3 = columnar.global !ht
  columnar.hj.build %2 : !buf -> %3 : !ht

  columnar.query.output %0, %1 : !columnar.col<si32>, !columnar.col<!columnar.str> ["r_regionkey", "r_name"]
}

```

## References
- https://voltrondata.com/blog/what-is-substrait-high-level-primer
- https://15721.courses.cs.cmu.edu/spring2018/papers/03-compilation/shaikhha-sigmod2016.pdf
- https://xuanwo.io/2024/02-what-i-talk-about-when-i-talk-about-query-optimizer-part-1/

On implementing hash tables/joins:
- https://15721.courses.cs.cmu.edu/spring2016/papers/p743-leis.pdf
- https://db.in.tum.de/~birler/papers/hashtable.pdf
