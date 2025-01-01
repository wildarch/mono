columnar.pipeline {
  %0 = columnar.constant #columnar.sel_id : !columnar.sel
  %1 = columnar.read_table "lineitem" "l_value" : <f64>
  columnar.print "l_value" %1 : <f64> sel=%0
}