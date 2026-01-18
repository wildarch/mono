// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<<i64, i32, !columnar.byte_array>>
columnar.global @buf : !buf init {
^bb0():
  %0 = columnar.type.size !columnar.struct<i64, i32, !columnar.byte_array>
  %1 = columnar.type.align !columnar.struct<i64, i32, !columnar.byte_array>
  %2 = columnar.runtime_call "col_tuple_buffer_init"(%0, %1) : (i64, i64) -> (!buf)
  columnar.global.return %2 : !buf
} destroy {
^bb0(%arg0 : !buf):
  columnar.runtime_call "col_tuple_buffer_destroy"(%arg0) : (!buf) -> ()
  columnar.global.return
}

// CHECK: columnar.pipeline_low global_open {
// CHECK:   %0 = columnar.constant_string "/host-home-folder/Downloads/tpch-sf1/region.parquet"
// CHECK:   %1 = columnar.runtime_call "col_table_scanner_open"(%0) : (!columnar.str_lit) -> !columnar.scanner_handle
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   %2 = columnar.runtime_call "col_table_column_open"(%0, %c0_i32) : (!columnar.str_lit, i32) -> !columnar.column_handle
// CHECK:   %c1_i32 = arith.constant 1 : i32
// CHECK:   %3 = columnar.runtime_call "col_table_column_open"(%0, %c1_i32) : (!columnar.str_lit, i32) -> !columnar.column_handle
// CHECK:   %4 = columnar.struct.alloc %1, %2, %3 : !columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle
// CHECK:   columnar.pipeline_low.yield %4 : !columnar.ptr<!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK: } local_open {
// CHECK: ^bb0(%arg0: !columnar.ptr<!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>):
// CHECK:   %0 = columnar.struct.get 0 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %1 = columnar.struct.get 1 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %2 = columnar.struct.get 2 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %3 = columnar.type.size !columnar.struct<i64, i32, !columnar.byte_array>
// CHECK:   %4 = columnar.type.align !columnar.struct<i64, i32, !columnar.byte_array>
// CHECK:   %5 = columnar.runtime_call "col_tuple_buffer_local_alloc"(%3, %4) : (i64, i64) -> !columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>
// CHECK:   %6 = columnar.struct.alloc %5 : !columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>
// CHECK:   columnar.pipeline_low.yield %6 : !columnar.ptr<!columnar.struct<!columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>>>
// CHECK: } body {
// CHECK: ^bb0(%arg0: !columnar.pipe_ctx, %arg1: !columnar.ptr<!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>, %arg2: !columnar.ptr<!columnar.struct<!columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>>>):
// CHECK:   %0 = columnar.struct.get 0 %arg1 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %1 = columnar.struct.get 1 %arg1 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %2 = columnar.struct.get 2 %arg1 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %3 = columnar.struct.get 0 %arg2 : <!columnar.struct<!columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>>>
// CHECK:   %4:3 = columnar.runtime_call "col_table_scanner_claim_chunk"(%0) : (!columnar.scanner_handle) -> (i32, i32, index)
// CHECK:   %c0 = arith.constant 0 : index
// CHECK:   %5 = arith.cmpi ugt, %4#2, %c0 : index
// %generated = Column read index vector (iota)
// CHECK:   %generated = tensor.generate %4#2 {
// CHECK:   ^bb0(%arg3: index):
// CHECK:     tensor.yield %arg3 : index
// CHECK:   } : tensor<?xindex>
// CHECK:   %6 = columnar.table.column.read %1 : tensor<?xsi32> row_group=%4#0 skip=%4#1 size=%4#2 ctx=%arg0
// CHECK:   %7 = columnar.table.column.read %2 : tensor<?x!columnar.byte_array> row_group=%4#0 skip=%4#1 size=%4#2 ctx=%arg0
// CHECK:   %c0_0 = arith.constant 0 : index
// CHECK:   %dim = tensor.dim %generated, %c0_0 : tensor<?xindex>
// %8 = hashed keys
// CHECK:   %8 = columnar.hash %6[%generated] : tensor<?xsi32> -> tensor<?xi64>
// %9 = insert locations for tuples
// CHECK:   %9 = columnar.tuple_buffer.insert %3, %8 : <<i64, i32, !columnar.byte_array>>
// CHECK:   %10 = columnar.runtime_call "col_tuple_buffer_local_get_allocator"(%3) : (!columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>) -> !columnar.allocator
// %generated_1 = insert locations for key
// CHECK:   %generated_1 = tensor.generate %dim {
// CHECK:   ^bb0(%arg3: index):
// CHECK:     %extracted = tensor.extract %9[%arg3] : tensor<?x!columnar.ptr<!columnar.struct<i64, i32, !columnar.byte_array>>>
// CHECK:     %11 = columnar.gfp %extracted 1 : <!columnar.struct<i64, i32, !columnar.byte_array>>
// CHECK:     tensor.yield %11 : !columnar.ptr<i32>
// CHECK:   } : tensor<?x!columnar.ptr<i32>>
// CHECK:   columnar.scatter %6[%generated] -> %generated_1 : tensor<?xsi32> -> tensor<?x!columnar.ptr<i32>> allocator=%10
// %generated_2 = insert locations for value
// CHECK:   %generated_2 = tensor.generate %dim {
// CHECK:   ^bb0(%arg3: index):
// CHECK:     %extracted = tensor.extract %9[%arg3] : tensor<?x!columnar.ptr<!columnar.struct<i64, i32, !columnar.byte_array>>>
// CHECK:     %11 = columnar.gfp %extracted 2 : <!columnar.struct<i64, i32, !columnar.byte_array>>
// CHECK:     tensor.yield %11 : !columnar.ptr<!columnar.byte_array>
// CHECK:   } : tensor<?x!columnar.ptr<!columnar.byte_array>>
// CHECK:   columnar.scatter %7[%generated] -> %generated_2 : tensor<?x!columnar.byte_array> -> tensor<?x!columnar.ptr<!columnar.byte_array>> allocator=%10
// CHECK:   columnar.pipeline_low.yield %5 : i1
// CHECK: } local_close {
// CHECK: ^bb0(%arg0: !columnar.ptr<!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>, %arg1: !columnar.ptr<!columnar.struct<!columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>>>):
// CHECK:   %0 = columnar.struct.get 0 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %1 = columnar.struct.get 1 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %2 = columnar.struct.get 2 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %3 = columnar.struct.get 0 %arg1 : <!columnar.struct<!columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>>>
// CHECK:   %4 = columnar.global.read @buf -> !columnar.tuple_buffer<<i64, i32, !columnar.byte_array>>
// CHECK:   columnar.runtime_call "col_tuple_buffer_merge"(%4, %3) : (!columnar.tuple_buffer<<i64, i32, !columnar.byte_array>>, !columnar.tuple_buffer.local<<i64, i32, !columnar.byte_array>>) -> ()
// CHECK:   columnar.pipeline_low.yield
// CHECK: } global_close {
// CHECK: ^bb0(%arg0: !columnar.ptr<!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>):
// CHECK:   %0 = columnar.struct.get 0 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %1 = columnar.struct.get 1 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   %2 = columnar.struct.get 2 %arg0 : <!columnar.struct<!columnar.scanner_handle, !columnar.column_handle, !columnar.column_handle>>
// CHECK:   columnar.pipeline_low.yield
// CHECK: }
columnar.pipeline {
  %sel, %r_regionkey, %r_name = columnar.read_table #table_region [
    #column_region_r_regionkey,
    #column_region_r_name] : !columnar.col<si32>, !columnar.col<!columnar.str>

  columnar.hj.collect
    keys=[%r_regionkey] : !columnar.col<si32>
    key_sel=[%sel]
    values=[%r_name] : !columnar.col<!columnar.str>
    value_sel=[%sel]
    -> @buf
}
