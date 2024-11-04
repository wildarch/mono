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

TODO: What do our operators look like?

## Query Optimization
Transformations to consider:
- Constant propagation: Becomes simpler because we always propagate per-column constant information.
- Push down predicates
- Join order optimization: Done exactly the same way as in regular relational algebra.
- Pull up projections: Easy to pull up individual columns
- Common sub expression elimination: Is largely the same as in the per-tuple case.

### Predicate Pushdown
TODO

### Projection Pull-up 
TODO

### Common Sub Expression Elimination
TODO

## Execution Plans

## References
- https://voltrondata.com/blog/what-is-substrait-high-level-primer
- https://15721.courses.cs.cmu.edu/spring2018/papers/03-compilation/shaikhha-sigmod2016.pdf
- https://xuanwo.io/2024/02-what-i-talk-about-when-i-talk-about-query-optimizer-part-1/