-- RUN: translate --data=%S/data --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#PARTKEY:]] = columnar.read_column #column_part_p_partkey : <si64>
-- CHECK: columnar.query.output %[[#PARTKEY]] {{.*}} ["partkey"]
SELECT p_partkey AS partkey
FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_part_p_partkey : <si64>
-- CHECK: %1 = columnar.read_column #column_part_p_name : <!columnar.str>
-- CHECK: %2 = columnar.read_column #column_part_p_mfgr : <!columnar.str>
-- CHECK: %3 = columnar.read_column #column_part_p_brand : <!columnar.str>
-- CHECK: %4 = columnar.read_column #column_part_p_type : <!columnar.str>
-- CHECK: %5 = columnar.read_column #column_part_p_size : <si32>
-- CHECK: %6 = columnar.read_column #column_part_p_container : <!columnar.str>
-- CHECK: %7 = columnar.read_column #column_part_p_retailprice : <!columnar.dec>
-- CHECK: %8 = columnar.read_column #column_part_p_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8
-- CHECK-SAME: ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]
SELECT * FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_supplier_s_suppkey : <si64>
-- CHECK: %1 = columnar.read_column #column_supplier_s_name : <!columnar.str>
-- CHECK: %2 = columnar.read_column #column_supplier_s_address : <!columnar.str>
-- CHECK: %3 = columnar.read_column #column_supplier_s_nationkey : <si32>
-- CHECK: %4 = columnar.read_column #column_supplier_s_phone : <!columnar.str>
-- CHECK: %5 = columnar.read_column #column_supplier_s_acctbal : <!columnar.dec>
-- CHECK: %6 = columnar.read_column #column_supplier_s_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6
-- CHECK-SAME: ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]
SELECT * FROM Supplier;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_partsupp_ps_partkey : <si64>
-- CHECK: %1 = columnar.read_column #column_partsupp_ps_suppkey : <si64>
-- CHECK: %2 = columnar.read_column #column_partsupp_ps_availqty : <si64>
-- CHECK: %3 = columnar.read_column #column_partsupp_ps_supplycost : <!columnar.dec>
-- CHECK: %4 = columnar.read_column #column_partsupp_ps_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4
-- CHECK-SAME: ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]
SELECT * FROM PartSupp;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_customer_c_custkey : <si64>
-- CHECK: %1 = columnar.read_column #column_customer_c_name : <!columnar.str>
-- CHECK: %2 = columnar.read_column #column_customer_c_address : <!columnar.str>
-- CHECK: %3 = columnar.read_column #column_customer_c_nationkey : <si32>
-- CHECK: %4 = columnar.read_column #column_customer_c_phone : <!columnar.str>
-- CHECK: %5 = columnar.read_column #column_customer_c_acctbal : <!columnar.dec>
-- CHECK: %6 = columnar.read_column #column_customer_c_mktsegment : <!columnar.str>
-- CHECK: %7 = columnar.read_column #column_customer_c_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7
-- CHECK-SAME: ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]
SELECT * FROM Customer;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_orders_o_orderkey : <si64>
-- CHECK: %1 = columnar.read_column #column_orders_o_custkey : <si64>
-- CHECK: %2 = columnar.read_column #column_orders_o_orderstatus : <!columnar.str>
-- CHECK: %3 = columnar.read_column #column_orders_o_totalprice : <!columnar.dec>
-- CHECK: %4 = columnar.read_column #column_orders_o_orderdate : <!columnar.date>
-- CHECK: %5 = columnar.read_column #column_orders_o_orderpriority : <!columnar.str>
-- CHECK: %6 = columnar.read_column #column_orders_o_clerk : <!columnar.str>
-- CHECK: %7 = columnar.read_column #column_orders_o_shippriority : <si32>
-- CHECK: %8 = columnar.read_column #column_orders_o_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8
-- CHECK-SAME: ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]
SELECT * FROM Orders;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_lineitem_l_orderkey : <si64>
-- CHECK: %1 = columnar.read_column #column_lineitem_l_partkey : <si64>
-- CHECK: %2 = columnar.read_column #column_lineitem_l_suppkey : <si64>
-- CHECK: %3 = columnar.read_column #column_lineitem_l_linenumber : <si64>
-- CHECK: %4 = columnar.read_column #column_lineitem_l_quantity : <!columnar.dec>
-- CHECK: %5 = columnar.read_column #column_lineitem_l_extendedprice : <!columnar.dec>
-- CHECK: %6 = columnar.read_column #column_lineitem_l_discount : <!columnar.dec>
-- CHECK: %7 = columnar.read_column #column_lineitem_l_tax : <!columnar.dec>
-- CHECK: %8 = columnar.read_column #column_lineitem_l_returnflag : <!columnar.str>
-- CHECK: %9 = columnar.read_column #column_lineitem_l_linestatus : <!columnar.str>
-- CHECK: %10 = columnar.read_column #column_lineitem_l_shipdate : <!columnar.date>
-- CHECK: %11 = columnar.read_column #column_lineitem_l_commitdate : <!columnar.date>
-- CHECK: %12 = columnar.read_column #column_lineitem_l_receiptdate : <!columnar.date>
-- CHECK: %13 = columnar.read_column #column_lineitem_l_shipinstruct : <!columnar.str>
-- CHECK: %14 = columnar.read_column #column_lineitem_l_shipmode : <!columnar.str>
-- CHECK: %15 = columnar.read_column #column_lineitem_l_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
-- CHECK-SAME: ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]
SELECT * FROM Lineitem;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_nation_n_nationkey : <si32>
-- CHECK: %1 = columnar.read_column #column_nation_n_name : <!columnar.str>
-- CHECK: %2 = columnar.read_column #column_nation_n_regionkey : <si32>
-- CHECK: %3 = columnar.read_column #column_nation_n_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3
-- CHECK-SAME: ["n_nationkey", "n_name", "n_regionkey", "n_comment"]
SELECT * FROM Nation;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_column #column_region_r_regionkey : <si32>
-- CHECK: %1 = columnar.read_column #column_region_r_name : <!columnar.str>
-- CHECK: %2 = columnar.read_column #column_region_r_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2
-- CHECK-SAME: ["r_regionkey", "r_name", "r_comment"]
SELECT * FROM Region;
