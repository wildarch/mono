// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<<i64, i32, !columnar.byte_array>>
columnar.global @buf : !buf init {
  %0 = columnar.type.size !columnar.struct<i64, i32, !columnar.byte_array>
  %1 = columnar.type.align !columnar.struct<i64, i32, !columnar.byte_array>
  %2 = columnar.runtime_call "col_tuple_buffer_init"(%0, %1) : (i64, i64) -> (!buf)
  columnar.global.return %2 : !buf
} destroy {
^bb0(%arg0 : !buf):
  columnar.runtime_call "col_tuple_buffer_destroy"(%arg0) : (!buf) -> ()
  columnar.global.return
}

!ht = !columnar.hash_table<(si32) -> (!columnar.str)>
columnar.global @ht : !ht init {
  %2 = columnar.runtime_call "col_hash_table_alloc"() : () -> (!ht)
  columnar.global.return %2 : !ht
} destroy {
^bb0(%arg0 : !ht):
  columnar.runtime_call "col_hash_table_destroy"(%arg0) : (!ht) -> ()
  columnar.global.return
}

columnar.pipeline {
  columnar.hj.build @buf -> @ht
}
