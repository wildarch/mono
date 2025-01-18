// RUN: mlir-opt --merge-chunks %s | FileCheck %s
#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>
module {
  columnar.pipeline {
    // CHECK: %[[#SEL_SCAN:]] = columnar.sel_scanner #table_mytable
    // CHECK: %[[#COL_SCAN:]] = columnar.open_column #column_mytable_l_value
    // CHECK: columnar.chunk {
    // CHECK:   %[[#SEL_CHUNK:]] = columnar.tensor.read_column %[[#SEL_SCAN]]
    // CHECK:   %[[#COL_CHUNK:]] = columnar.tensor.read_column %[[#COL_SCAN]]
    // CHECK:   columnar.tensor.print "l_value" %[[#COL_CHUNK]]
    // CHECK-SAME: sel=%[[#SEL_CHUNK]]
    // CHECK:   columnar.chunk.yield
    %0 = columnar.sel_scanner #table_mytable
    %1 = columnar.chunk -> tensor<?xindex> {
      %4 = columnar.tensor.read_column %0 : tensor<?xindex>
      columnar.chunk.yield %4 : tensor<?xindex>
    }
    %2 = columnar.open_column #column_mytable_l_value
    %3 = columnar.chunk -> tensor<?xf64> {
      %4 = columnar.tensor.read_column %2 : tensor<?xf64>
      columnar.chunk.yield %4 : tensor<?xf64>
    }
    columnar.chunk %3, %1 : tensor<?xf64>, tensor<?xindex> {
    ^bb0(%arg0: tensor<?xf64>, %arg1: tensor<?xindex>):
      columnar.tensor.print "l_value" %arg0 : tensor<?xf64> sel=%arg1 : tensor<?xindex>
      columnar.chunk.yield
    }
  }
}