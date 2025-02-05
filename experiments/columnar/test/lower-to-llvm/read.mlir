!globals = !columnar.ptr<!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.print_handle>>

columnar.pipeline_low global_open {
  %c3 = arith.constant 3 : index
  %c42 = arith.constant 42 : index
  %0 = columnar.runtime_call "col_table_scanner_open"(%c42) : (index) -> !columnar.scanner_handle
  %1 = columnar.runtime_call "col_table_column_open"(%c42, %c3) : (index, index) -> !columnar.column_handle
  %2 = columnar.runtime_call "col_print_open"() : () -> !columnar.print_handle
  %3 = columnar.struct.alloc %0, %1, %2 : !columnar.scanner_handle, !columnar.column_handle, !columnar.print_handle
  columnar.pipeline_low.yield %3 : !globals
} body {
^bb0(%arg0: !globals):
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = columnar.struct.get %arg0 : !globals 0
  %1 = columnar.struct.get %arg0 : !globals 1
  %2 = columnar.struct.get %arg0 : !globals 2
  %3:2 = columnar.runtime_call "col_table_scanner_claim_chunk"(%0) : (!columnar.scanner_handle) -> (index, index)
  %4 = arith.cmpi ugt, %3#1, %c0 : index
  %alloc = memref.alloc(%3#1) {alignment = 64 : i64} : memref<?xindex>
  scf.for %arg1 = %c0 to %3#1 step %c1 {
    memref.store %arg1, %alloc[%arg1] : memref<?xindex>
  }
  %alloc_0 = memref.alloc(%3#1) : memref<?xf64>
  columnar.runtime_call "col_table_column_read"(%1, %3#0, %3#1, %alloc_0) : (!columnar.column_handle, index, index, memref<?xf64>) -> ()
  %5 = columnar.runtime_call "col_print_chunk_alloc"(%3#1) : (index) -> !columnar.print_chunk
  columnar.runtime_call "col_print_chunk_append"(%5, %alloc_0, %alloc) : (!columnar.print_chunk, memref<?xf64>, memref<?xindex>) -> ()
  columnar.runtime_call "col_print_write"(%2, %5) : (!columnar.print_handle, !columnar.print_chunk) -> ()
  columnar.pipeline_low.yield %4 : i1
} global_close {
^bb0(%arg0: !globals):
  columnar.pipeline_low.yield
}