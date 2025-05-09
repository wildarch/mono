-- RUN: translate --data=%S/data --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#PRICE:]] = columnar.read_column #column_part_p_retailprice
-- CHECK: %[[#AGG:]] = columnar.aggregate aggregate %[[#PRICE]] {{.*}} [SUM]
-- CHECK: columnar.query.output %[[#AGG]] {{.*}} ["p_retailprice"]
SELECT SUM(p_retailprice)
FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#BRAND:]] = columnar.read_column #column_part_p_brand
-- CHECK: %[[#PRICE:]] = columnar.read_column #column_part_p_retailprice
-- CHECK: %[[#AGG:]]:2 = columnar.aggregate group %[[#BRAND]]
-- CHECK-SAME: aggregate %[[#PRICE]] {{.*}} [SUM]
-- CHECK: columnar.query.output %[[#AGG]]#0, %[[#AGG]]#1 : !columnar.col<!columnar.str>, !columnar.col<!columnar.dec> ["p_brand", "p_retailprice"]
SELECT p_brand, SUM(p_retailprice)
FROM Part
GROUP BY p_brand;
