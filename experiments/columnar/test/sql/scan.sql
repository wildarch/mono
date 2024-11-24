-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK: %[[#PARTKEY:]] = columnar.read_table "part" "p_partkey" : <i64>
-- CHECK: columnar.query.output %[[#PARTKEY]] {{.*}} ["partkey"]
SELECT p_partkey AS partkey
FROM Part;