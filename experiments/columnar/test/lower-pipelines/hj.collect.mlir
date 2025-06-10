// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<i64, si32, !columnar.str>
%buf = columnar.global !buf

columnar.pipeline {
  %sel, %r_regionkey, %r_name = columnar.read_table #table_region [
    #column_region_r_regionkey,
    #column_region_r_name] : !columnar.col<si32>, !columnar.col<!columnar.str>

  // TODO: use selection vector
  columnar.hj.collect
    keys=[%r_regionkey] : !columnar.col<si32>
    key_sel=[%sel]
    values=[%r_name] : !columnar.col<!columnar.str>
    value_sel=[%sel]
    -> %buf : !buf
}
