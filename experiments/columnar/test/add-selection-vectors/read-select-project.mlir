module {
  columnar.query {
    %0 = columnar.read_table "lineitem" "l_quantity" : <!columnar.dec>
    %1 = columnar.read_table "lineitem" "l_extendedprice" : <!columnar.dec>
    %2 = columnar.read_table "lineitem" "l_discount" : <!columnar.dec>
    %3 = columnar.read_table "lineitem" "l_shipdate" : <!columnar.date>
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
}