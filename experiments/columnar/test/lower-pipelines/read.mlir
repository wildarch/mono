// RUN: mlir-opt --lower-pipelines %s | FileCheck %s
#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>

// CHECK: columnar.pipeline_low 
// CHECK-LABEL: global_open {
// CHECK: %[[#SEL_SCAN:]] = columnar.table.scanner.open #table_mytable
// CHECK: %[[#COL_SCAN:]] = columnar.table.column.open #column_mytable_l_value
// CHECK: columnar.pipeline_low.yield %[[#SEL_SCAN]], %[[#COL_SCAN]]
//
// CHECK-LABEL: body {
// CHECK: %start, %size = columnar.table.scanner.claim_chunk %arg0
// CHECK: %c0 = arith.constant 0
// CHECK: %[[#MORE:]] = arith.cmpi ugt, %size, %c0
// CHECK: %generated = tensor.generate %size
// CHECK:   tensor.yield %arg
// CHECK: %[[#COL:]] = columnar.table.column.read %arg1 : tensor<?xf64> %start %size
// CHECK: columnar.tensor.print "l_value" %[[#COL]] : tensor<?xf64> sel=%generated : tensor<?xindex>
// CHECK: columnar.pipeline_low.yield %[[#MORE]] : i1
columnar.pipeline {
  %sel, %col = columnar.read_table #table_mytable [#column_mytable_l_value] : !columnar.col<f64>
  columnar.print ["l_value"] %col : !columnar.col<f64> sel=%sel
}