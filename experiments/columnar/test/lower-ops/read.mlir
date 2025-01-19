// RUN: mlir-opt --lower-ops %s | FileCheck %s
#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>
columnar.pipeline {
  %0 = columnar.sel_scanner #table_mytable
  %1 = columnar.open_column #column_mytable_l_value
  columnar.chunk {
    %2 = columnar.tensor.read_column %0 : tensor<?xindex>
    %3 = columnar.tensor.read_column %1 : tensor<?xf64>
    columnar.tensor.print "l_value" %3 : tensor<?xf64> sel=%2 : tensor<?xindex>
    columnar.chunk.yield
  }
}