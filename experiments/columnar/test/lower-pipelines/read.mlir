// RUN: mlir-opt --lower-pipelines %s | FileCheck %s
#table_mytable = #columnar.table<"mytable" path="/home/daan/Downloads/tpch-sf1/nation.tab">
#column_mytable_l_value = #columnar.table_col<#table_mytable "n_nationkey" : i32 path="/home/daan/Downloads/tpch-sf1/n_nationkey.col"> 

// CHECK: columnar.pipeline_low 
// CHECK-LABEL: global_open {
// CHECK: %[[#SEL_SCAN:]] = columnar.table.scanner.open #table_mytable
// CHECK: %[[#COL_SCAN:]] = columnar.table.column.open #column_mytable_l_value
// CHECK: %[[#PRINT:]] = columnar.print.open
// CHECK: columnar.pipeline_low.yield %[[#SEL_SCAN]], %[[#COL_SCAN]], %[[#PRINT]]
//
// CHECK-LABEL: body {
// CHECK: %start, %size = columnar.table.scanner.claim_chunk %arg0
// CHECK: %c0 = arith.constant 0
// CHECK: %[[#MORE:]] = arith.cmpi ugt, %size, %c0
// CHECK: %generated = tensor.generate %size
// CHECK:   tensor.yield %arg
// CHECK: %[[#COL:]] = columnar.table.column.read %arg1 : tensor<?xf64> %start %size
//
// CHECK: %dim = tensor.dim %generated, %c0
// CHECK: %[[#CHUNK:]] = columnar.print.chunk.alloc %dim
// CHECK: columnar.print.chunk.append %[[#CHUNK]] 
// CHECK-SAME: %[[#COL]] 
// CHECK-SAME: sel=%generated
// CHECK: columnar.print.write %arg2 %[[#CHUNK]]
//
// CHECK: columnar.pipeline_low.yield %[[#MORE]] : i1
columnar.pipeline {
  %sel, %col = columnar.read_table #table_mytable [#column_mytable_l_value] : !columnar.col<i32>
  columnar.print ["n_nationkey"] %col : !columnar.col<i32> sel=%sel
}