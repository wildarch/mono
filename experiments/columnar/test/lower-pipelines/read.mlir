#table_mytable = #columnar.table<"mytable">
#column_mytable_l_value = #columnar.table_col<#table_mytable "l_value" : f64>

columnar.pipeline {
  %0 = columnar.sel.table #table_mytable
  %1 = columnar.read_table #column_mytable_l_value : <f64>
  columnar.print "l_value" %1 : <f64> sel=%0
}