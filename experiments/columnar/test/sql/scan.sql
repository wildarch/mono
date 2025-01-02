-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK: #table_customer = #columnar.table<"customer">
-- CHECK: #table_lineitem = #columnar.table<"lineitem">
-- CHECK: #table_nation = #columnar.table<"nation">
-- CHECK: #table_orders = #columnar.table<"orders">
-- CHECK: #table_part = #columnar.table<"part">
-- CHECK: #table_partsupp = #columnar.table<"partsupp">
-- CHECK: #table_region = #columnar.table<"region">
-- CHECK: #table_supplier = #columnar.table<"supplier">
-- CHECK: #column_customer_c_acctbal = #columnar.table_col<#table_customer "c_acctbal" : !columnar.dec>
-- CHECK: #column_customer_c_address = #columnar.table_col<#table_customer "c_address" : !columnar.str>
-- CHECK: #column_customer_c_comment = #columnar.table_col<#table_customer "c_comment" : !columnar.str>
-- CHECK: #column_customer_c_custkey = #columnar.table_col<#table_customer "c_custkey" : i64>
-- CHECK: #column_customer_c_mktsegment = #columnar.table_col<#table_customer "c_mktsegment" : !columnar.str>
-- CHECK: #column_customer_c_name = #columnar.table_col<#table_customer "c_name" : !columnar.str>
-- CHECK: #column_customer_c_nationkey = #columnar.table_col<#table_customer "c_nationkey" : i64>
-- CHECK: #column_customer_c_phone = #columnar.table_col<#table_customer "c_phone" : !columnar.str>
-- CHECK: #column_lineitem_l_comment = #columnar.table_col<#table_lineitem "l_comment" : !columnar.str>
-- CHECK: #column_lineitem_l_commitdate = #columnar.table_col<#table_lineitem "l_commitdate" : !columnar.date>
-- CHECK: #column_lineitem_l_discount = #columnar.table_col<#table_lineitem "l_discount" : !columnar.dec>
-- CHECK: #column_lineitem_l_extendedprice = #columnar.table_col<#table_lineitem "l_extendedprice" : !columnar.dec>
-- CHECK: #column_lineitem_l_linenumber = #columnar.table_col<#table_lineitem "l_linenumber" : si64>
-- CHECK: #column_lineitem_l_linestatus = #columnar.table_col<#table_lineitem "l_linestatus" : !columnar.str>
-- CHECK: #column_lineitem_l_orderkey = #columnar.table_col<#table_lineitem "l_orderkey" : i64>
-- CHECK: #column_lineitem_l_partkey = #columnar.table_col<#table_lineitem "l_partkey" : i64>
-- CHECK: #column_lineitem_l_quantity = #columnar.table_col<#table_lineitem "l_quantity" : !columnar.dec>
-- CHECK: #column_lineitem_l_receiptdate = #columnar.table_col<#table_lineitem "l_receiptdate" : !columnar.date>
-- CHECK: #column_lineitem_l_returnflag = #columnar.table_col<#table_lineitem "l_returnflag" : !columnar.str>
-- CHECK: #column_lineitem_l_shipdate = #columnar.table_col<#table_lineitem "l_shipdate" : !columnar.date>
-- CHECK: #column_lineitem_l_shipinstruct = #columnar.table_col<#table_lineitem "l_shipinstruct" : !columnar.str>
-- CHECK: #column_lineitem_l_shipmode = #columnar.table_col<#table_lineitem "l_shipmode" : !columnar.str>
-- CHECK: #column_lineitem_l_suppkey = #columnar.table_col<#table_lineitem "l_suppkey" : i64>
-- CHECK: #column_lineitem_l_tax = #columnar.table_col<#table_lineitem "l_tax" : !columnar.dec>
-- CHECK: #column_nation_n_comment = #columnar.table_col<#table_nation "n_comment" : !columnar.str>
-- CHECK: #column_nation_n_name = #columnar.table_col<#table_nation "n_name" : !columnar.str>
-- CHECK: #column_nation_n_nationkey = #columnar.table_col<#table_nation "n_nationkey" : i64>
-- CHECK: #column_nation_n_regionkey = #columnar.table_col<#table_nation "n_regionkey" : i64>
-- CHECK: #column_orders_o_clerk = #columnar.table_col<#table_orders "o_clerk" : !columnar.str>
-- CHECK: #column_orders_o_comment = #columnar.table_col<#table_orders "o_comment" : !columnar.str>
-- CHECK: #column_orders_o_custkey = #columnar.table_col<#table_orders "o_custkey" : i64>
-- CHECK: #column_orders_o_orderdate = #columnar.table_col<#table_orders "o_orderdate" : !columnar.date>
-- CHECK: #column_orders_o_orderkey = #columnar.table_col<#table_orders "o_orderkey" : i64>
-- CHECK: #column_orders_o_orderpriority = #columnar.table_col<#table_orders "o_orderpriority" : !columnar.str>
-- CHECK: #column_orders_o_orderstatus = #columnar.table_col<#table_orders "o_orderstatus" : !columnar.str>
-- CHECK: #column_orders_o_shippriority = #columnar.table_col<#table_orders "o_shippriority" : si64>
-- CHECK: #column_orders_o_totalprice = #columnar.table_col<#table_orders "o_totalprice" : !columnar.dec>
-- CHECK: #column_part_p_brand = #columnar.table_col<#table_part "p_brand" : !columnar.str>
-- CHECK: #column_part_p_comment = #columnar.table_col<#table_part "p_comment" : !columnar.str>
-- CHECK: #column_part_p_container = #columnar.table_col<#table_part "p_container" : !columnar.str>
-- CHECK: #column_part_p_mfgr = #columnar.table_col<#table_part "p_mfgr" : !columnar.str>
-- CHECK: #column_part_p_name = #columnar.table_col<#table_part "p_name" : !columnar.str>
-- CHECK: #column_part_p_partkey = #columnar.table_col<#table_part "p_partkey" : i64>
-- CHECK: #column_part_p_retailprice = #columnar.table_col<#table_part "p_retailprice" : !columnar.dec>
-- CHECK: #column_part_p_size = #columnar.table_col<#table_part "p_size" : si64>
-- CHECK: #column_part_p_type = #columnar.table_col<#table_part "p_type" : !columnar.str>
-- CHECK: #column_partsupp_ps_availqty = #columnar.table_col<#table_partsupp "ps_availqty" : si64>
-- CHECK: #column_partsupp_ps_comment = #columnar.table_col<#table_partsupp "ps_comment" : !columnar.str>
-- CHECK: #column_partsupp_ps_partkey = #columnar.table_col<#table_partsupp "ps_partkey" : i64>
-- CHECK: #column_partsupp_ps_suppkey = #columnar.table_col<#table_partsupp "ps_suppkey" : i64>
-- CHECK: #column_partsupp_ps_supplycost = #columnar.table_col<#table_partsupp "ps_supplycost" : !columnar.dec>
-- CHECK: #column_region_r_comment = #columnar.table_col<#table_region "r_comment" : !columnar.str>
-- CHECK: #column_region_r_name = #columnar.table_col<#table_region "r_name" : !columnar.str>
-- CHECK: #column_region_r_regionkey = #columnar.table_col<#table_region "r_regionkey" : i64>
-- CHECK: #column_supplier_s_acctbal = #columnar.table_col<#table_supplier "s_acctbal" : !columnar.dec>
-- CHECK: #column_supplier_s_address = #columnar.table_col<#table_supplier "s_address" : !columnar.str>
-- CHECK: #column_supplier_s_comment = #columnar.table_col<#table_supplier "s_comment" : !columnar.str>
-- CHECK: #column_supplier_s_name = #columnar.table_col<#table_supplier "s_name" : !columnar.str>
-- CHECK: #column_supplier_s_nationkey = #columnar.table_col<#table_supplier "s_nationkey" : i64>
-- CHECK: #column_supplier_s_phone = #columnar.table_col<#table_supplier "s_phone" : !columnar.str>
-- CHECK: #column_supplier_s_suppkey = #columnar.table_col<#table_supplier "s_suppkey" : i64>

-- CHECK-LABEL: columnar.query {
-- CHECK: %[[#PARTKEY:]] = columnar.read_table #column_part_p_partkey : <i64>
-- CHECK: columnar.query.output %[[#PARTKEY]] {{.*}} ["partkey"]
SELECT p_partkey AS partkey
FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_part_p_partkey : <i64>
-- CHECK: %1 = columnar.read_table #column_part_p_name : <!columnar.str>
-- CHECK: %2 = columnar.read_table #column_part_p_mfgr : <!columnar.str>
-- CHECK: %3 = columnar.read_table #column_part_p_brand : <!columnar.str>
-- CHECK: %4 = columnar.read_table #column_part_p_type : <!columnar.str>
-- CHECK: %5 = columnar.read_table #column_part_p_size : <si64>
-- CHECK: %6 = columnar.read_table #column_part_p_container : <!columnar.str>
-- CHECK: %7 = columnar.read_table #column_part_p_retailprice : <!columnar.dec>
-- CHECK: %8 = columnar.read_table #column_part_p_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8
-- CHECK-SAME: ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]
SELECT * FROM Part;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_supplier_s_suppkey : <i64>
-- CHECK: %1 = columnar.read_table #column_supplier_s_name : <!columnar.str>
-- CHECK: %2 = columnar.read_table #column_supplier_s_address : <!columnar.str>
-- CHECK: %3 = columnar.read_table #column_supplier_s_nationkey : <i64>
-- CHECK: %4 = columnar.read_table #column_supplier_s_phone : <!columnar.str>
-- CHECK: %5 = columnar.read_table #column_supplier_s_acctbal : <!columnar.dec>
-- CHECK: %6 = columnar.read_table #column_supplier_s_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6
-- CHECK-SAME: ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]
SELECT * FROM Supplier;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_partsupp_ps_partkey : <i64>
-- CHECK: %1 = columnar.read_table #column_partsupp_ps_suppkey : <i64>
-- CHECK: %2 = columnar.read_table #column_partsupp_ps_availqty : <si64>
-- CHECK: %3 = columnar.read_table #column_partsupp_ps_supplycost : <!columnar.dec>
-- CHECK: %4 = columnar.read_table #column_partsupp_ps_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4
-- CHECK-SAME: ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]
SELECT * FROM PartSupp;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_customer_c_custkey : <i64>
-- CHECK: %1 = columnar.read_table #column_customer_c_name : <!columnar.str>
-- CHECK: %2 = columnar.read_table #column_customer_c_address : <!columnar.str>
-- CHECK: %3 = columnar.read_table #column_customer_c_nationkey : <i64>
-- CHECK: %4 = columnar.read_table #column_customer_c_phone : <!columnar.str>
-- CHECK: %5 = columnar.read_table #column_customer_c_acctbal : <!columnar.dec>
-- CHECK: %6 = columnar.read_table #column_customer_c_mktsegment : <!columnar.str>
-- CHECK: %7 = columnar.read_table #column_customer_c_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7
-- CHECK-SAME: ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]
SELECT * FROM Customer;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_orders_o_orderkey : <i64>
-- CHECK: %1 = columnar.read_table #column_orders_o_custkey : <i64>
-- CHECK: %2 = columnar.read_table #column_orders_o_orderstatus : <!columnar.str>
-- CHECK: %3 = columnar.read_table #column_orders_o_totalprice : <!columnar.dec>
-- CHECK: %4 = columnar.read_table #column_orders_o_orderdate : <!columnar.date>
-- CHECK: %5 = columnar.read_table #column_orders_o_orderpriority : <!columnar.str>
-- CHECK: %6 = columnar.read_table #column_orders_o_clerk : <!columnar.str>
-- CHECK: %7 = columnar.read_table #column_orders_o_shippriority : <si64>
-- CHECK: %8 = columnar.read_table #column_orders_o_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8
-- CHECK-SAME: ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]
SELECT * FROM Orders;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_lineitem_l_orderkey : <i64>
-- CHECK: %1 = columnar.read_table #column_lineitem_l_partkey : <i64>
-- CHECK: %2 = columnar.read_table #column_lineitem_l_suppkey : <i64>
-- CHECK: %3 = columnar.read_table #column_lineitem_l_linenumber : <si64>
-- CHECK: %4 = columnar.read_table #column_lineitem_l_quantity : <!columnar.dec>
-- CHECK: %5 = columnar.read_table #column_lineitem_l_extendedprice : <!columnar.dec>
-- CHECK: %6 = columnar.read_table #column_lineitem_l_discount : <!columnar.dec>
-- CHECK: %7 = columnar.read_table #column_lineitem_l_tax : <!columnar.dec>
-- CHECK: %8 = columnar.read_table #column_lineitem_l_returnflag : <!columnar.str>
-- CHECK: %9 = columnar.read_table #column_lineitem_l_linestatus : <!columnar.str>
-- CHECK: %10 = columnar.read_table #column_lineitem_l_shipdate : <!columnar.date>
-- CHECK: %11 = columnar.read_table #column_lineitem_l_commitdate : <!columnar.date>
-- CHECK: %12 = columnar.read_table #column_lineitem_l_receiptdate : <!columnar.date>
-- CHECK: %13 = columnar.read_table #column_lineitem_l_shipinstruct : <!columnar.str>
-- CHECK: %14 = columnar.read_table #column_lineitem_l_shipmode : <!columnar.str>
-- CHECK: %15 = columnar.read_table #column_lineitem_l_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
-- CHECK-SAME: ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]
SELECT * FROM Lineitem;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_nation_n_nationkey : <i64>
-- CHECK: %1 = columnar.read_table #column_nation_n_name : <!columnar.str>
-- CHECK: %2 = columnar.read_table #column_nation_n_regionkey : <i64>
-- CHECK: %3 = columnar.read_table #column_nation_n_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2, %3
-- CHECK-SAME: ["n_nationkey", "n_name", "n_regionkey", "n_comment"]
SELECT * FROM Nation;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.read_table #column_region_r_regionkey : <i64>
-- CHECK: %1 = columnar.read_table #column_region_r_name : <!columnar.str>
-- CHECK: %2 = columnar.read_table #column_region_r_comment : <!columnar.str>
-- CHECK: columnar.query.output %0, %1, %2
-- CHECK-SAME: ["r_regionkey", "r_name", "r_comment"]
SELECT * FROM Region;