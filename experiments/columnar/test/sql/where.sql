-- RUN: translate --import-sql %s | mlir-opt --push-down-predicates | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#ORDERKEY:]] = columnar.read_table "lineitem" "l_orderkey"
-- CHECK: %[[#LINENUMBER:]] = columnar.read_table "lineitem" "l_linenumber"
-- CHECK: %[[#SELECT:]]:2 = columnar.select %[[#ORDERKEY]], %[[#LINENUMBER]]
-- CHECK:   %[[#CONST:]] = columnar.constant 24
-- CHECK:   %[[#CMP:]] = columnar.cmp LT %arg1, %[[#CONST]]
-- CHECK:   columnar.select.return %[[#CMP]]
-- CHECK: columnar.query.output %[[#SELECT]]#0 {{.*}} ["l_orderkey"]
SELECT l_orderkey
FROM lineitem
WHERE l_linenumber < 24;

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#ORDERKEY:]] = columnar.read_table "lineitem" "l_orderkey"
-- CHECK: %[[#QTY:]] = columnar.read_table "lineitem" "l_quantity"
-- CHECK: %[[#SELECT:]]:2 = columnar.select %[[#ORDERKEY]], %[[#QTY]]
-- CHECK:   %[[#CONST:]] = columnar.constant #columnar<dec 2400>
-- CHECK:   %[[#CMP:]] = columnar.cmp LT %arg1, %[[#CONST]]
-- CHECK:   columnar.select.return %[[#CMP]]
-- CHECK: columnar.query.output %[[#SELECT]]#0 : !columnar.col<i64> ["l_orderkey"]
SELECT l_orderkey
FROM lineitem
WHERE l_quantity < 24;

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#SHIPDATE:]] = columnar.read_table "lineitem" "l_shipdate" : <!columnar.date>
-- CHECK: %[[#SELECT:]] = columnar.select %[[#SHIPDATE]]
-- CHECK:   %[[#CONST:]] = columnar.constant #columnar<date 1994 1 1>
-- CHECK:   %[[#CMP:]] = columnar.cmp GE %arg0, %[[#CONST]]
-- CHECK:   columnar.select.return %[[#CMP]]
-- CHECK: columnar.query.output %[[#SELECT]] {{.*}} ["l_shipdate"]
SELECT l_shipdate
FROM lineitem
WHERE l_shipdate >= CAST('1994-01-01' AS date);