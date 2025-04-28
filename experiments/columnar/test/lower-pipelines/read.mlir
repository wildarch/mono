// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_nation = #columnar.table<"nation" path="experiments/columnar/test/sql/data/nation.parquet">
#column_nation_n_nationkey = #columnar.table_col<#table_nation 0 "n_nationkey" : si32[i32]>

// CHECK: columnar.pipeline_low
// CHECK-LABEL: global_open {
// CHECK: %[[#NATION_PATH:]] = columnar.constant_string "experiments/columnar/test/sql/data/nation.parquet"
// CHECK: %[[#SCAN_OPEN:]] = columnar.runtime_call "col_table_scanner_open"(%[[#NATION_PATH]]) : (!columnar.str_lit) -> !columnar.scanner_handle
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[#COL_OPEN:]] = columnar.runtime_call "col_table_column_open"(%[[#NATION_PATH]], %c0_i32) : (!columnar.str_lit, i32) -> !columnar.column_handle
// CHECK: %[[#PRINT_OPEN:]] = columnar.runtime_call "col_print_open"() : () -> !columnar.print_handle
// CHECK: %[[#GLOBALS:]] = columnar.struct.alloc %[[#SCAN_OPEN]], %[[#COL_OPEN]], %[[#PRINT_OPEN]]
// CHECK: columnar.pipeline_low.yield %[[#GLOBALS]]
//
// CHECK-LABEL: body {
// CHECK: %[[#SCAN:]] = columnar.struct.get 0 %arg0
// CHECK: %[[#COL:]] = columnar.struct.get 1 %arg0
// CHECK: %[[#PRINT:]] = columnar.struct.get 2 %arg0
// CHECK: %[[#CLAIM:]]:3 = columnar.runtime_call "col_table_scanner_claim_chunk"(%[[#SCAN]])
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %[[#MORE:]] = arith.cmpi ugt, %[[#CLAIM]]#2, %c0
// CHECK: %generated = tensor.generate %[[#CLAIM]]#2 {
// CHECK:   tensor.yield %arg1
// CHECK: %[[#READ:]] = columnar.table.column.read %[[#COL]] : tensor<?xi32> %[[#CLAIM]]#0 %[[#CLAIM]]#1 %[[#CLAIM]]#2
// CHECK: %c0_0 = arith.constant 0 : index
// CHECK: %dim = tensor.dim %generated, %c0_0 : tensor<?xindex>
// CHECK: %[[#CHUNK:]] = columnar.runtime_call "col_print_chunk_alloc"(%dim) : (index) -> !columnar.print_chunk
// CHECK: columnar.print.chunk.append %[[#CHUNK]] %[[#READ]]
// CHECK-SAME: sel=%generated
// CHECK: columnar.runtime_call "col_print_write"(%[[#PRINT]], %[[#CHUNK]])
// CHECK: columnar.pipeline_low.yield %[[#MORE]]

columnar.pipeline {
  %sel, %col = columnar.read_table #table_nation [#column_nation_n_nationkey] : !columnar.col<i32>
  columnar.query.output %col : !columnar.col<i32> ["n_nationkey"] sel=%sel
}
