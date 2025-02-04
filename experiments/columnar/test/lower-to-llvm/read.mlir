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
  %alloc = memref.alloc(%size) {alignment = 64 : i64} : memref<?xindex>
  linalg.map outs(%alloc : memref<?xindex>)
      () {
      %2 = linalg.index 0 : index
      linalg.yield %2 : index
    }
  %alloc_0 = memref.alloc(%size) : memref<?xf64>
  columnar.runtime_call "col_table_column_read"(%arg1, %start, %size, %alloc_0) : (!columnar.column_handle, index, index, memref<?xf64>) -> ()
  %1 = columnar.print.chunk.alloc %size
  columnar.runtime_call "col_print_chunk_append"(%1, %alloc_0, %alloc) : (!columnar.print_chunk, memref<?xf64>, memref<?xindex>) -> ()
  columnar.print.write %arg2 %1
  columnar.pipeline_low.yield %0 : i1
} global_close {
^bb0(%arg0: !columnar.scanner_handle, %arg1: !columnar.column_handle, %arg2: !columnar.print_handle):
  columnar.pipeline_low.yield
}