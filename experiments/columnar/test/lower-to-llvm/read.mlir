columnar.pipeline_low global_open {
  %c42 = arith.constant 42 : index
  %0 = columnar.runtime_call "col_table_scanner_open"(%c42) : (index) -> (!columnar.scanner_handle)
  %c3 = arith.constant 3 : index
  %1 = columnar.runtime_call "col_table_column_open"(%c42, %c3) : (index, index) -> (!columnar.column_handle)
  %2 = columnar.runtime_call "col_print_open"() : () -> (!columnar.print_handle)
  columnar.pipeline_low.yield %0, %1, %2 : !columnar.scanner_handle, !columnar.column_handle, !columnar.print_handle
} body {
^bb0(%arg0: !columnar.scanner_handle, %arg1: !columnar.column_handle, %arg2: !columnar.print_handle):
  %c0 = arith.constant 0 : index
  // TODO: avoid multiple return values.
  %start, %size = columnar.runtime_call "col_table_scanner_claim_chunk"(%arg0) : (!columnar.scanner_handle) -> (index, index)
  %0 = arith.cmpi ugt, %size, %c0 : index
  %alloc = memref.alloc(%size) {alignment = 64 : i64} : memref<?xindex>
  linalg.map outs(%alloc : memref<?xindex>)
      () {
      %2 = linalg.index 0 : index
      linalg.yield %2 : index
    }
  %alloc_0 = memref.alloc(%size) : memref<?xf64>
  columnar.runtime_call "col_table_column_read"(%arg1, %start, %size, %alloc_0) : (!columnar.column_handle, index, index, memref<?xf64>) -> ()
  %1 = columnar.runtime_call "col_print_chunk_alloc"(%size) : (index) -> (!columnar.print_chunk)
  columnar.runtime_call "col_print_chunk_append"(%1, %alloc_0, %alloc) : (!columnar.print_chunk, memref<?xf64>, memref<?xindex>) -> ()
  columnar.runtime_call "col_print_write"(%arg2, %1) : (!columnar.print_handle, !columnar.print_chunk) -> ()
  columnar.pipeline_low.yield %0 : i1
} global_close {
^bb0(%arg0: !columnar.scanner_handle, %arg1: !columnar.column_handle, %arg2: !columnar.print_handle):
  columnar.pipeline_low.yield
}