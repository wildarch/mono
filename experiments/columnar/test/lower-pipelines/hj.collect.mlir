// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<<i64, i32, !columnar.byte_array>>
columnar.global @buf !buf

columnar.pipeline {
  %sel, %r_regionkey, %r_name = columnar.read_table #table_region [
    #column_region_r_regionkey,
    #column_region_r_name] : !columnar.col<si32>, !columnar.col<!columnar.str>

  // CHECK:%[[#BUFFER:]] = columnar.struct.get 0 %arg2 : <!columnar.struct<!columnar.tuple_buffer.local<i64, si32, !columnar.str parts=16>>>
  //
  // CHECK: %[[#REGIONKEY:]] = columnar.table.column.read
  // CHECK: %[[#NAME:]] = columnar.table.column.read
  // CHECK: %dim = tensor.dim %generated, %c0
  // CHECK: %generated_1 = tensor.generate %dim
  // CHECK:   %c0_i64 = arith.constant 0 : i64
  // CHECK:   tensor.yield %c0_i64
  // CHECK: %[[#HASH:]] = columnar.hash %generated_1, %[[#REGIONKEY]][%generated]
  // CHECK: %[[#INSERT:]] = columnar.tuple_buffer.insert %[[#BUFFER]], %9 : <i64, si32, !columnar.str parts=16>
  // CHECK: %c0_2 = arith.constant 0 : index
  // CHECK: %[[#SCATTER_REGIONKEY:]] = columnar.scatter %[[#REGIONKEY]][%generated] -> %[[#INSERT]][%c0_2]
  // CHECK: %[[#SCATTER_NAME:]] = columnar.scatter %[[#NAME]][%generated] -> %[[#INSERT]][%[[#SCATTER_REGIONKEY]]]
  columnar.hj.collect
    keys=[%r_regionkey] : !columnar.col<si32>
    key_sel=[%sel]
    values=[%r_name] : !columnar.col<!columnar.str>
    value_sel=[%sel]
    -> @buf
}
