// RUN: mlir-opt --lower-pipelines %s | FileCheck %s
#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>

// CHECK: columnar.pipeline_low 
// CHECK-LABEL: global_open {
// CHECK: %[[#SEL_SCAN:]] = columnar.sel_scanner #table_mytable
// CHECK: %[[#COL_SCAN:]] = columnar.open_column #column_mytable_l_value
// CHECK: columnar.pipeline_low.yield %[[#SEL_SCAN]], %[[#COL_SCAN]]
//
// CHECK-LABEL: body {
// CHECK: %[[#SEL:]] = columnar.tensor.read_column %arg0 : tensor<?xindex>
// CHECK: %[[#SEL_MORE:]] = columnar.scanner.have_more %arg0
// CHECK: %[[#COL:]] = columnar.tensor.read_column %arg1 : tensor<?xf64>
// CHECK: %[[#COL_MORE:]] = columnar.scanner.have_more %arg1
// CHECK: columnar.tensor.print "l_value" %[[#COL]] : tensor<?xf64> sel=%[[#SEL]] : tensor<?xindex>
// CHECK: %[[#ALL_MORE:]] = arith.andi %[[#SEL_MORE]], %[[#COL_MORE]]
// CHECK: columnar.pipeline_low.yield %[[#ALL_MORE]]
columnar.pipeline {
  %0 = columnar.sel.table #table_mytable
  %1 = columnar.read_table #column_mytable_l_value : <f64>
  columnar.print ["l_value"] %1 : !columnar.col<f64> sel=%0
}