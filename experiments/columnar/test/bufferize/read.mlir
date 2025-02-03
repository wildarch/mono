#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>

columnar.pipeline_low global_open {
  %0 = columnar.table.scanner.open #table_mytable
  %1 = columnar.table.column.open #column_mytable_l_value
  %2 = columnar.print.open
  columnar.pipeline_low.yield %0, %1, %2 : !columnar.scanner_handle, !columnar.column_handle, !columnar.print_handle
} body {
^bb0(%arg0: !columnar.scanner_handle, %arg1: !columnar.column_handle, %arg2: !columnar.print_handle):
  %c0 = arith.constant 0 : index
  %start, %size = columnar.table.scanner.claim_chunk %arg0
  %0 = arith.cmpi ugt, %size, %c0 : index
  %generated = tensor.generate %size {
  ^bb0(%arg3: index):
    tensor.yield %arg3 : index
  } : tensor<?xindex>
  %1 = columnar.table.column.read %arg1 : tensor<?xf64> %start %size
  %2 = columnar.print.chunk.alloc %size
  columnar.print.chunk.append %2 %1 : tensor<?xf64> sel=%generated : tensor<?xindex>
  columnar.print.write %arg2 %2
  columnar.pipeline_low.yield %0 : i1
} global_close {
^bb0(%arg0: !columnar.scanner_handle, %arg1: !columnar.column_handle, %arg2: !columnar.print_handle):
  columnar.pipeline_low.yield
}