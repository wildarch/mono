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

## Lowering
Steps:
1. Make selection vectors explicit (`select` is removed)
2. Make nulls explicit
3. Split pipeline breakers into source/sink side (`join`, `aggregate` and `union`)
4. Group into pipelines
5. Lower pipeline to `func` over `memref`. Body uses `arith` and `vector`.

## References
- https://voltrondata.com/blog/what-is-substrait-high-level-primer
- https://15721.courses.cs.cmu.edu/spring2018/papers/03-compilation/shaikhha-sigmod2016.pdf
- https://xuanwo.io/2024/02-what-i-talk-about-when-i-talk-about-query-optimizer-part-1/