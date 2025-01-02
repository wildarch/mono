-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_region_r_regionkey
-- CHECK: %1 = columnar.read_table #column_region_r_name
-- CHECK: %2 = columnar.read_table #column_region_r_comment
-- CHECK: %3:3 = columnar.order_by %1 : !columnar.col<!columnar.str> [ASC] %0, %2
-- CHECK: columnar.query.output %3#1, %3#0, %3#2
-- CHECK-SAME: ["r_regionkey", "r_name", "r_comment"]
SELECT * FROM Region
ORDER BY r_name;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_region_r_regionkey
-- CHECK: %1 = columnar.read_table #column_region_r_name
-- CHECK: %2 = columnar.read_table #column_region_r_comment
-- CHECK: %3:3 = columnar.order_by %1 : !columnar.col<!columnar.str> [DESC] %0, %2
-- CHECK: columnar.query.output %3#1, %3#0, %3#2
-- CHECK-SAME: ["r_regionkey", "r_name", "r_comment"]
SELECT * FROM Region
ORDER BY r_name DESC;