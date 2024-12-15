-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#PARTKEY:]] = columnar.read_table "part" "p_partkey" : <i64>
-- CHECK: columnar.query.output %[[#PARTKEY]] {{.*}} ["partkey"]
SELECT p_partkey AS partkey
FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "part" "p_partkey" : <i64>
-- CHECK: %1 = columnar.read_table "part" "p_name" : <!columnar.str>
-- CHECK: %2 = columnar.read_table "part" "p_mfgr" : <!columnar.str>
-- CHECK: %3 = columnar.read_table "part" "p_brand" : <!columnar.str>
-- CHECK: %4 = columnar.read_table "part" "p_type" : <!columnar.str>
-- CHECK: %5 = columnar.read_table "part" "p_size" : <si64>
-- CHECK: %6 = columnar.read_table "part" "p_container" : <!columnar.str>
-- CHECK: %7 = columnar.read_table "part" "p_retailprice" : <!columnar.dec>
-- CHECK: %8 = columnar.read_table "part" "p_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8
-- CHECK-SAME: ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]
SELECT * FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "supplier" "s_suppkey" : <i64>
-- CHECK: %1 = columnar.read_table "supplier" "s_name" : <!columnar.str>
-- CHECK: %2 = columnar.read_table "supplier" "s_address" : <!columnar.str>
-- CHECK: %3 = columnar.read_table "supplier" "s_nationkey" : <i64>
-- CHECK: %4 = columnar.read_table "supplier" "s_phone" : <!columnar.str>
-- CHECK: %5 = columnar.read_table "supplier" "s_acctbal" : <!columnar.dec>
-- CHECK: %6 = columnar.read_table "supplier" "s_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6
-- CHECK-SAME: ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]
SELECT * FROM Supplier;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "partsupp" "ps_partkey" : <i64>
-- CHECK: %1 = columnar.read_table "partsupp" "ps_suppkey" : <i64>
-- CHECK: %2 = columnar.read_table "partsupp" "ps_availqty" : <si64>
-- CHECK: %3 = columnar.read_table "partsupp" "ps_supplycost" : <!columnar.dec>
-- CHECK: %4 = columnar.read_table "partsupp" "ps_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4
-- CHECK-SAME: ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]
SELECT * FROM PartSupp;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "customer" "c_custkey" : <i64>
-- CHECK: %1 = columnar.read_table "customer" "c_name" : <!columnar.str>
-- CHECK: %2 = columnar.read_table "customer" "c_address" : <!columnar.str>
-- CHECK: %3 = columnar.read_table "customer" "c_nationkey" : <i64>
-- CHECK: %4 = columnar.read_table "customer" "c_phone" : <!columnar.str>
-- CHECK: %5 = columnar.read_table "customer" "c_acctbal" : <!columnar.dec>
-- CHECK: %6 = columnar.read_table "customer" "c_mktsegment" : <!columnar.str>
-- CHECK: %7 = columnar.read_table "customer" "c_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7
-- CHECK-SAME: ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]
SELECT * FROM Customer;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "orders" "o_orderkey" : <i64>
-- CHECK: %1 = columnar.read_table "orders" "o_custkey" : <i64>
-- CHECK: %2 = columnar.read_table "orders" "o_orderstatus" : <!columnar.str>
-- CHECK: %3 = columnar.read_table "orders" "o_totalprice" : <!columnar.dec>
-- CHECK: %4 = columnar.read_table "orders" "o_orderdate" : <!columnar.date>
-- CHECK: %5 = columnar.read_table "orders" "o_orderpriority" : <!columnar.str>
-- CHECK: %6 = columnar.read_table "orders" "o_clerk" : <!columnar.str>
-- CHECK: %7 = columnar.read_table "orders" "o_shippriority" : <si64>
-- CHECK: %8 = columnar.read_table "orders" "o_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8
-- CHECK-SAME: ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]
SELECT * FROM Orders;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "lineitem" "l_orderkey" : <i64>
-- CHECK: %1 = columnar.read_table "lineitem" "l_partkey" : <i64>
-- CHECK: %2 = columnar.read_table "lineitem" "l_suppkey" : <i64>
-- CHECK: %3 = columnar.read_table "lineitem" "l_linenumber" : <si64>
-- CHECK: %4 = columnar.read_table "lineitem" "l_quantity" : <!columnar.dec>
-- CHECK: %5 = columnar.read_table "lineitem" "l_extendedprice" : <!columnar.dec>
-- CHECK: %6 = columnar.read_table "lineitem" "l_discount" : <!columnar.dec>
-- CHECK: %7 = columnar.read_table "lineitem" "l_tax" : <!columnar.dec>
-- CHECK: %8 = columnar.read_table "lineitem" "l_returnflag" : <!columnar.str>
-- CHECK: %9 = columnar.read_table "lineitem" "l_linestatus" : <!columnar.str>
-- CHECK: %10 = columnar.read_table "lineitem" "l_shipdate" : <!columnar.date>
-- CHECK: %11 = columnar.read_table "lineitem" "l_commitdate" : <!columnar.date>
-- CHECK: %12 = columnar.read_table "lineitem" "l_receiptdate" : <!columnar.date>
-- CHECK: %13 = columnar.read_table "lineitem" "l_shipinstruct" : <!columnar.str>
-- CHECK: %14 = columnar.read_table "lineitem" "l_shipmode" : <!columnar.str>
-- CHECK: %15 = columnar.read_table "lineitem" "l_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
-- CHECK-SAME: ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]
SELECT * FROM Lineitem;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "nation" "n_nationkey" : <i64>
-- CHECK: %1 = columnar.read_table "nation" "n_name" : <!columnar.str>
-- CHECK: %2 = columnar.read_table "nation" "n_regionkey" : <i64>
-- CHECK: %3 = columnar.read_table "nation" "n_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3
-- CHECK-SAME: ["n_nationkey", "n_name", "n_regionkey", "n_comment"]
SELECT * FROM Nation;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table "region" "r_regionkey" : <i64>
-- CHECK: %1 = columnar.read_table "region" "r_name" : <!columnar.str>
-- CHECK: %2 = columnar.read_table "region" "r_comment" : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2
-- CHECK-SAME: ["r_regionkey", "r_name", "r_comment"]
SELECT * FROM Region;