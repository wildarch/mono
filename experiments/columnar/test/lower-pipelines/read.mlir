// RUN: mlir-opt --lower-pipelines %s | FileCheck %s
#table_mytable = #columnar.table<"mytable" path="/home/daan/Downloads/tpch-sf1/nation.tab">
#column_mytable_l_value = #columnar.table_col<#table_mytable "n_nationkey" : i32 path="/home/daan/Downloads/tpch-sf1/n_nationkey.col"> 

// CHECK: columnar.pipeline_low 
// CHECK-LABEL: global_open {
// CHECK: %[[#NATION_PATH:]] = columnar.constant_string "/home/daan/Downloads/tpch-sf1/nation.tab"
// CHECK: %[[#SCAN_OPEN:]] = columnar.runtime_call "col_table_scanner_open"(%[[#NATION_PATH]]) : (!columnar.str_lit) -> !columnar.scanner_handle
// CHECK: %[[#NATIONKEY_PATH:]] = columnar.constant_string "/home/daan/Downloads/tpch-sf1/n_nationkey.col"
// CHECK: %[[#COL_OPEN:]] = columnar.runtime_call "col_table_column_open"(%[[#NATIONKEY_PATH]]) : (!columnar.str_lit) -> !columnar.column_handle
// CHECK: %[[#PRINT_OPEN:]] = columnar.runtime_call "col_print_open"() : () -> !columnar.print_handle
// CHECK: %[[#GLOBALS:]] = columnar.struct.alloc %[[#SCAN_OPEN]], %[[#COL_OPEN]], %[[#PRINT_OPEN]]
// CHECK: columnar.pipeline_low.yield %[[#GLOBALS]]
//
// CHECK-LABEL: body {
// CHECK: %[[#SCAN:]] = columnar.struct.get 0 %arg0
// CHECK: %[[#COL:]] = columnar.struct.get 1 %arg0
// CHECK: %[[#PRINT:]] = columnar.struct.get 2 %arg0
// CHECK: %[[#CLAIM:]]:2 = columnar.runtime_call "col_table_scanner_claim_chunk"(%[[#SCAN]])
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %[[#MORE:]] = arith.cmpi ugt, %[[#CLAIM]]#1, %c0
// CHECK: %generated = tensor.generate %[[#CLAIM]]#1 {
// CHECK:   tensor.yield %arg1
// CHECK: %[[#READ:]] = columnar.table.column.read %[[#COL]] : tensor<?xi32> %[[#CLAIM]]#0 %[[#CLAIM]]#1
// CHECK: %c0_0 = arith.constant 0 : index
// CHECK: %dim = tensor.dim %generated, %c0_0 : tensor<?xindex>
// CHECK: %[[#CHUNK:]] = columnar.runtime_call "col_print_chunk_alloc"(%dim) : (index) -> !columnar.print_chunk
// CHECK: columnar.print.chunk.append %[[#CHUNK]] %[[#READ]] 
// CHECK-SAME: sel=%generated
// CHECK: columnar.runtime_call "col_print_write"(%[[#PRINT]], %[[#CHUNK]])
// CHECK: columnar.pipeline_low.yield %[[#MORE]]

columnar.pipeline {
  %sel, %col = columnar.read_table #table_mytable [#column_mytable_l_value] : !columnar.col<i32>
  columnar.print ["n_nationkey"] %col : !columnar.col<i32> sel=%sel
}