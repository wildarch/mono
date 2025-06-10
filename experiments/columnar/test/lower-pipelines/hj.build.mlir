// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<i64, si32, !columnar.str>
%buf = columnar.global !buf

!ht = !columnar.hash_table<(si32) -> (!columnar.str)>
%ht = columnar.global !ht

columnar.pipeline {
  columnar.hj.build %buf : !buf -> %ht : !ht
}
