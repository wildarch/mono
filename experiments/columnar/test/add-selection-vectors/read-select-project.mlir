// RUN: columnar-opt --add-selection-vectors %s | FileCheck %s

#table_lineitem = #columnar.table<"lineitem" path="/tmp/lineitem.tab">
#column_lineitem_l_quantity = #columnar.table_col<#table_lineitem "l_quantity" : !columnar.dec>
#column_lineitem_l_extendedprice = #columnar.table_col<#table_lineitem "l_extendedprice" : !columnar.dec>
#column_lineitem_l_discount = #columnar.table_col<#table_lineitem "l_discount" : !columnar.dec>
#column_lineitem_l_shipdate = #columnar.table_col<#table_lineitem "l_shipdate" : !columnar.date>

columnar.query {
  // CHECK: %0 = columnar.constant #columnar<dec 2400> : !columnar.dec
  // CHECK: %1 = columnar.constant #columnar<dec 7> : !columnar.dec
  // CHECK: %2 = columnar.constant #columnar<dec 5> : !columnar.dec
  // CHECK: %3 = columnar.constant #columnar<date 1995 1 1> : !columnar.date
  // CHECK: %4 = columnar.constant #columnar<date 1994 1 1> : !columnar.date
  // CHECK: %5 = columnar.sel.table #table_lineitem
  // CHECK: %6 = columnar.read_column #column_lineitem_l_quantity : <!columnar.dec>
  // CHECK: %7 = columnar.read_column #column_lineitem_l_extendedprice : <!columnar.dec>
  // CHECK: %8 = columnar.read_column #column_lineitem_l_discount : <!columnar.dec>
  // CHECK: %9 = columnar.read_column #column_lineitem_l_shipdate : <!columnar.date>
  // CHECK: %10 = columnar.cmp GE %9, %4 : <!columnar.date> sel=%5
  // CHECK: %11 = columnar.sel.filter %5 by %10 filter_sel=%5
  // CHECK: %12 = columnar.cmp LT %9, %3 : <!columnar.date> sel=%11
  // CHECK: %13 = columnar.sel.filter %11 by %12 filter_sel=%11
  // CHECK: %14 = columnar.cmp LE %2, %8 : <!columnar.dec> sel=%13
  // CHECK: %15 = columnar.sel.filter %13 by %14 filter_sel=%13
  // CHECK: %16 = columnar.cmp LE %8, %1 : <!columnar.dec> sel=%15
  // CHECK: %17 = columnar.sel.filter %15 by %16 filter_sel=%15
  // CHECK: %18 = columnar.cmp LT %6, %0 : <!columnar.dec> sel=%17
  // CHECK: %19 = columnar.sel.filter %17 by %18 filter_sel=%17
  // CHECK: %20 = columnar.mul %7, %8 : !columnar.col<!columnar.dec> sel=%19
  // CHECK: columnar.query.output %9, %20 : !columnar.col<!columnar.date>, !columnar.col<!columnar.dec> ["l_shipdate", "revenue"] sel=%19
  %0 = columnar.read_column #column_lineitem_l_quantity : <!columnar.dec>
  %1 = columnar.read_column #column_lineitem_l_extendedprice : <!columnar.dec>
  %2 = columnar.read_column #column_lineitem_l_discount : <!columnar.dec>
  %3 = columnar.read_column #column_lineitem_l_shipdate : <!columnar.date>
  %4:4 = columnar.select %0, %1, %2, %3 : !columnar.col<!columnar.dec>, !columnar.col<!columnar.dec>, !columnar.col<!columnar.dec>, !columnar.col<!columnar.date> {
  ^bb0(%arg0: !columnar.col<!columnar.dec>, %arg1: !columnar.col<!columnar.dec>, %arg2: !columnar.col<!columnar.dec>, %arg3: !columnar.col<!columnar.date>):
    columnar.pred %arg3 : !columnar.col<!columnar.date> {
    ^bb0(%arg4: !columnar.col<!columnar.date>):
      %6 = columnar.constant #columnar<date 1994 1 1> : !columnar.date
      %7 = columnar.cmp GE %arg4, %6 : <!columnar.date>
      columnar.pred.eval %7
    }
    columnar.pred %arg3 : !columnar.col<!columnar.date> {
    ^bb0(%arg4: !columnar.col<!columnar.date>):
      %6 = columnar.constant #columnar<date 1995 1 1> : !columnar.date
      %7 = columnar.cmp LT %arg4, %6 : <!columnar.date>
      columnar.pred.eval %7
    }
    columnar.pred %arg2 : !columnar.col<!columnar.dec> {
    ^bb0(%arg4: !columnar.col<!columnar.dec>):
      %6 = columnar.constant #columnar<dec 5> : !columnar.dec
      %7 = columnar.cmp LE %6, %arg4 : <!columnar.dec>
      columnar.pred.eval %7
    }
    columnar.pred %arg2 : !columnar.col<!columnar.dec> {
    ^bb0(%arg4: !columnar.col<!columnar.dec>):
      %6 = columnar.constant #columnar<dec 7> : !columnar.dec
      %7 = columnar.cmp LE %arg4, %6 : <!columnar.dec>
      columnar.pred.eval %7
    }
    columnar.pred %arg0 : !columnar.col<!columnar.dec> {
    ^bb0(%arg4: !columnar.col<!columnar.dec>):
      %6 = columnar.constant #columnar<dec 2400> : !columnar.dec
      %7 = columnar.cmp LT %arg4, %6 : <!columnar.dec>
      columnar.pred.eval %7
    }
  }
  %5 = columnar.mul %4#1, %4#2 : !columnar.col<!columnar.dec>
  columnar.query.output %4#3, %5 : !columnar.col<!columnar.date>, !columnar.col<!columnar.dec> ["l_shipdate", "revenue"]
}
