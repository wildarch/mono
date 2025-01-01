columnar.pipeline {
  %0 = columnar.constant #columnar<dec 2400> : !columnar.dec
  %1 = columnar.constant #columnar<dec 7> : !columnar.dec
  %2 = columnar.constant #columnar<dec 5> : !columnar.dec
  %3 = columnar.constant #columnar<date 1995 1 1> : !columnar.date
  %4 = columnar.constant #columnar<date 1994 1 1> : !columnar.date
  %5 = columnar.constant #columnar.sel_id : !columnar.sel
  %6 = columnar.read_table "lineitem" "l_quantity" : <!columnar.dec>
  %7 = columnar.read_table "lineitem" "l_extendedprice" : <!columnar.dec>
  %8 = columnar.read_table "lineitem" "l_discount" : <!columnar.dec>
  %9 = columnar.read_table "lineitem" "l_shipdate" : <!columnar.date>
  %10 = columnar.cmp GE %9, %4 : <!columnar.date> sel=%5
  %11 = columnar.sel.filter %5 by %10 filter_sel=%5
  %12 = columnar.cmp LT %9, %3 : <!columnar.date> sel=%11
  %13 = columnar.sel.filter %11 by %12 filter_sel=%11
  %14 = columnar.cmp LE %2, %8 : <!columnar.dec> sel=%13
  %15 = columnar.sel.filter %13 by %14 filter_sel=%13
  %16 = columnar.cmp LE %8, %1 : <!columnar.dec> sel=%15
  %17 = columnar.sel.filter %15 by %16 filter_sel=%15
  %18 = columnar.cmp LT %6, %0 : <!columnar.dec> sel=%17
  %19 = columnar.sel.filter %17 by %18 filter_sel=%17
  %20 = columnar.mul %7, %8 : !columnar.col<!columnar.dec> sel=%19
  columnar.print "l_shipdate" %9 : <!columnar.date> sel=%19
  columnar.print "revenue" %20 : <!columnar.dec> sel=%19
}