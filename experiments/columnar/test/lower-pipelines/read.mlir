// RUN: mlir-opt --lower-pipelines %s | FileCheck %s
#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>

columnar.pipeline {
  // CHECK: %[[#SEL_SCAN:]] = columnar.sel_scanner #table_mytable
  // CHECK: %[[#SEL_CHUNK:]] = columnar.chunk
  // CHECK:   %[[#READ:]] = columnar.tensor.read_column %[[#SEL_SCAN]] : tensor<?xindex>
  // CHECK:   columnar.chunk.yield %[[#READ]]
  // CHECK: }
  %0 = columnar.sel.table #table_mytable
  // CHECK: %[[#COL_SCAN:]] = columnar.open_column #column_mytable_l_value
  // CHECK: %[[#COL_CHUNK:]] = columnar.chunk
  // CHECK:   %[[#READ:]] = columnar.tensor.read_column %[[#COL_SCAN]] : tensor<?xf64>
  // CHECK:   columnar.chunk.yield %[[#READ]]
  // CHECK: }
  %1 = columnar.read_table #column_mytable_l_value : <f64>
  // CHECK: columnar.chunk %[[#COL_CHUNK]], %[[#SEL_CHUNK]]
  // CHECK:   columnar.tensor.print "l_value" %arg0
  // CHECK-SAME: sel=%arg1
  // CHECK:   columnar.chunk.yield
  columnar.print "l_value" %1 : <f64> sel=%0
}