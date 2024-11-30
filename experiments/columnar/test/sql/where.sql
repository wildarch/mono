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
-- CHECK:   %[[#CONST:]] = columnar.constant 24
-- CHECK:   %[[#CAST:]] = columnar.cast %[[#CONST]] : <si64> -> <!columnar.dec>
-- CHECK:   %[[#CMP:]] = columnar.cmp LT %arg1, %4
-- CHECK:   columnar.select.return %[[#CMP]]
-- CHECK: columnar.query.output %[[#SELECT]]#0 : !columnar.col<i64> ["l_orderkey"]
SELECT l_orderkey
FROM lineitem
WHERE l_quantity < 24;