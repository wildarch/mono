// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_nation = #columnar.table<"nation" path="/host-home-folder/Downloads/tpch-sf1/nation.parquet">
#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_nation_n_name = #columnar.table_col<#table_nation 1 "n_name" : !columnar.str[!columnar.byte_array]>
#column_nation_n_regionkey = #columnar.table_col<#table_nation 2 "n_regionkey" : si32[i32]>
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!ht = !columnar.hash_table<(si32) -> (!columnar.str)>
%ht = columnar.global !ht

// TODO: depend on build
columnar.pipeline {
  %sel, %n_regionkey, %n_name = columnar.read_table #table_nation [
    #column_nation_n_regionkey,
    #column_nation_n_name] : !columnar.col<si32>, !columnar.col<!columnar.str>

  // TODO: Use input selection vector
  %r_name, %r_sel = columnar.hj.probe %n_regionkey : !columnar.col<si32>
    probe %ht : !ht

  // TODO: use selection vector
  columnar.query.output %r_name, %n_name
    : !columnar.col<!columnar.str>
    , !columnar.col<!columnar.str> ["r_name", "n_name"]
}
