// RUN: columnar-opt --lower-pipelines %s | FileCheck %s

#table_nation = #columnar.table<"nation" path="/host-home-folder/Downloads/tpch-sf1/nation.parquet">
#table_region = #columnar.table<"region" path="/host-home-folder/Downloads/tpch-sf1/region.parquet">
#column_nation_n_name = #columnar.table_col<#table_nation 1 "n_name" : !columnar.str[!columnar.byte_array]>
#column_nation_n_regionkey = #columnar.table_col<#table_nation 2 "n_regionkey" : si32[i32]>
#column_region_r_name = #columnar.table_col<#table_region 1 "r_name" : !columnar.str[!columnar.byte_array]>
#column_region_r_regionkey = #columnar.table_col<#table_region 0 "r_regionkey" : si32[i32]>

!buf = !columnar.tuple_buffer<i64, si32, !columnar.str>
!ht = !columnar.hash_table<(si32) -> (!columnar.str)>

columnar.query {
  %0 = columnar.read_column #column_region_r_regionkey : <si32>
  %1 = columnar.read_column #column_region_r_name : <!columnar.str>
  %2 = columnar.read_column #column_nation_n_name : <!columnar.str>
  %3 = columnar.read_column #column_nation_n_regionkey : <si32>
  %4:4 = columnar.join (%0, %1) (%2, %3) : (!columnar.col<si32>, !columnar.col<!columnar.str>) (!columnar.col<!columnar.str>, !columnar.col<si32>)
  %5:4 = columnar.select %4#0, %4#1, %4#2, %4#3 : !columnar.col<si32>, !columnar.col<!columnar.str>, !columnar.col<!columnar.str>, !columnar.col<si32> {
  ^bb0(%arg0: !columnar.col<si32>, %arg1: !columnar.col<!columnar.str>, %arg2: !columnar.col<!columnar.str>, %arg3: !columnar.col<si32>):
    columnar.pred %arg0, %arg3 : !columnar.col<si32>, !columnar.col<si32> {
    ^bb0(%arg4: !columnar.col<si32>, %arg5: !columnar.col<si32>):
      %6 = columnar.cmp EQ %arg5, %arg4 : <si32>
      columnar.pred.eval %6
    }
  }
  columnar.query.output %5#1, %5#2 : !columnar.col<!columnar.str>, !columnar.col<!columnar.str> ["r_name", "n_name"]
}

%buf = columnar.global !buf

columnar.pipeline {
  %sel, %r_regionkey, %r_name = columnar.read_table #table_region [
    #column_region_r_regionkey,
    #column_region_r_name] : !columnar.col<si32>, !columnar.col<!columnar.str>

  // TODO: use selection vector
  columnar.hj.collect
    keys=[%r_regionkey] : !columnar.col<si32>
    values=[%r_name] : !columnar.col<!columnar.str>
    -> %buf : !buf
}

%ht = columnar.global !ht

// TODO: depend on collect
columnar.pipeline {
  columnar.hj.build %buf : !buf -> %ht : !ht
}

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
