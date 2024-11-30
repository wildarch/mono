-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "nation" "n_nationkey"
-- CHECK: %1 = columnar.read_table "nation" "n_name"
-- CHECK: %2 = columnar.read_table "nation" "n_regionkey"
-- CHECK: %3 = columnar.read_table "nation" "n_comment"
-- CHECK: %4 = columnar.read_table "region" "r_regionkey"
-- CHECK: %5 = columnar.read_table "region" "r_name"
-- CHECK: %6 = columnar.read_table "region" "r_comment"
-- CHECK: %7:7 = columnar.join (%0, %1, %2, %3) (%4, %5, %6)
-- CHECK: columnar.query.output %7#1, %7#5 {{.*}} ["n_name", "r_name"]
SELECT n_name, r_name
FROM Nation, Region;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "nation" "n_nationkey"
-- CHECK: %1 = columnar.read_table "nation" "n_name"
-- CHECK: %2 = columnar.read_table "nation" "n_regionkey"
-- CHECK: %3 = columnar.read_table "nation" "n_comment"
-- CHECK: %4 = columnar.read_table "partsupp" "ps_partkey"
-- CHECK: %5 = columnar.read_table "partsupp" "ps_suppkey"
-- CHECK: %6 = columnar.read_table "partsupp" "ps_availqty"
-- CHECK: %7 = columnar.read_table "partsupp" "ps_supplycost"
-- CHECK: %8 = columnar.read_table "partsupp" "ps_comment"
-- CHECK: %9:9 = columnar.join (%0, %1, %2, %3) (%4, %5, %6, %7, %8)
-- CHECK: %10 = columnar.read_table "region" "r_regionkey"
-- CHECK: %11 = columnar.read_table "region" "r_name"
-- CHECK: %12 = columnar.read_table "region" "r_comment"
-- CHECK: %13:12 = columnar.join (%9#0, %9#1, %9#2, %9#3, %9#4, %9#5, %9#6, %9#7, %9#8) (%10, %11, %12)
-- CHECK: columnar.query.output %13#1, %13#8, %13#10 
-- CHECK-SAME: ["n_name", "ps_comment", "r_name"]
SELECT n_name, ps_comment, r_name
FROM Nation, PartSupp, Region;