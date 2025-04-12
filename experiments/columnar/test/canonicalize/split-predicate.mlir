// RUN columnar-opt --canonicalize %s | FileCheck %s

#table_nation = #columnar.table<"nation" path="/tmp/nation.tab">
#table_region = #columnar.table<"region" path="/tmp/region.tab">
#column_nation_n_comment = #columnar.table_col<#table_nation "n_comment" : !columnar.str path="/tmp/n_comment.col">
#column_nation_n_name = #columnar.table_col<#table_nation "n_name" : !columnar.str path="/tmp/n_name.col">
#column_nation_n_nationkey = #columnar.table_col<#table_nation "n_nationkey" : i64 path="/tmp/n_nationkey.col">
#column_nation_n_regionkey = #columnar.table_col<#table_nation "n_regionkey" : i64 path="/tmp/n_regionkey.col">
#column_region_r_comment = #columnar.table_col<#table_region "r_comment" : !columnar.str path="/tmp/r_comment.col">
#column_region_r_name = #columnar.table_col<#table_region "r_name" : !columnar.str path="/tmp/r_name.col">
#column_region_r_regionkey = #columnar.table_col<#table_region "r_regionkey" : i64 path="/tmp/r_regionkey.col">

columnar.query {
  %0 = columnar.read_column #column_nation_n_nationkey : <i64>
  %1 = columnar.read_column #column_nation_n_name : <!columnar.str>
  %2 = columnar.read_column #column_nation_n_regionkey : <i64>
  %3 = columnar.read_column #column_nation_n_comment : <!columnar.str>
  %4 = columnar.read_column #column_region_r_regionkey : <i64>
  %5 = columnar.read_column #column_region_r_name : <!columnar.str>
  %6 = columnar.read_column #column_region_r_comment : <!columnar.str>
  %7:7 = columnar.join (%0, %1, %2, %3) (%4, %5, %6) : (!columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<i64>, !columnar.col<!columnar.str>) (!columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<!columnar.str>)
  %8:7 = columnar.select %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6 : !columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<!columnar.str> {
  ^bb0(%arg0: !columnar.col<i64>, %arg1: !columnar.col<!columnar.str>, %arg2: !columnar.col<i64>, %arg3: !columnar.col<!columnar.str>, %arg4: !columnar.col<i64>, %arg5: !columnar.col<!columnar.str>, %arg6: !columnar.col<!columnar.str>):
    // CHECK-NOT: columnar.pred
    // CHECK: columnar.pred %arg2, %arg4
    // CHECK: ^bb0(%arg7: !columnar.col<i64>, %arg8: !columnar.col<i64>):
    // CHECK:   %9 = columnar.cmp EQ %arg7, %arg8 : <i64>
    // CHECK:   columnar.pred.eval %9
    //
    // CHECK: columnar.pred %arg1
    // CHECK: ^bb0(%arg7: !columnar.col<!columnar.str>):
    // CHECK:   %9 = columnar.constant #columnar<str "Netherlands"> : !columnar.str
    // CHECK:   %10 = columnar.cmp EQ %arg7, %9 : <!columnar.str>
    // CHECK:   columnar.pred.eval %10
    // CHECK-NOT: columnar.pred
    columnar.pred %arg1, %arg2, %arg4 : !columnar.col<!columnar.str>, !columnar.col<i64>, !columnar.col<i64> {
    ^bb0(%arg7: !columnar.col<!columnar.str>, %arg8: !columnar.col<i64>, %arg9: !columnar.col<i64>):
      %9 = columnar.cmp EQ %arg8, %arg9 : <i64>
      %10 = columnar.constant #columnar<str "Netherlands"> : !columnar.str
      %11 = columnar.cmp EQ %arg7, %10 : <!columnar.str>
      %12 = columnar.and %9, %11
      columnar.pred.eval %12
    }
  }
  columnar.query.output %8#0, %8#1, %8#2, %8#3, %8#4, %8#5, %8#6 : !columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<i64>, !columnar.col<!columnar.str>, !columnar.col<!columnar.str> ["n_nationkey", "n_name", "n_regionkey", "n_comment", "r_regionkey", "r_name", "r_comment"]
}
