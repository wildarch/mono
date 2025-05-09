// RUN: columnar-opt --add-selection-vectors %s | FileCheck %s

#table_nation = #columnar.table<"nation" path="experiments/columnar/test/sql/data/nation.parquet">
#column_nation_n_nationkey = #columnar.table_col<#table_nation 0 "n_nationkey" : si32[i32]>

columnar.query {
  // CHECK: %[[#SEL:]] = columnar.sel.table #table_nation
  // CHECK: %[[#READ:]] = columnar.read_column
  %0 = columnar.read_column #column_nation_n_nationkey : <i64>

  // CHECK: columnar.query.output %[[#READ]]
  // CHECK-SAME: sel=%[[#SEL]]
  columnar.query.output %0 : !columnar.col<i64> ["n_nationkey"]
}
