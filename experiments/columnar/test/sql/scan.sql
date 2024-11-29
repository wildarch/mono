-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK: %[[#PARTKEY:]] = columnar.read_table "part" "p_partkey" : <i64>
-- CHECK: columnar.query.output %[[#PARTKEY]] {{.*}} ["partkey"]
SELECT p_partkey AS partkey
FROM Part;

SELECT * FROM Part;
-- SELECT * FROM Supplier;
-- SELECT * FROM PartSupp;
-- SELECT * FROM Customer;
-- SELECT * FROM Orders;
-- SELECT * FROM Lineitem;
-- SELECT * FROM Nation;
-- SELECT * FROM Region;