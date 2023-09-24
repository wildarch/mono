%scan = operator.scan_parquet 
    "/home/daan/workspace/duckdb/duckdb_benchmark_data/tpch_sf1_parquet/lineitem.parquet" 
    ["l_orderkey", "l_shipdate"] : table<i32, i32>
%1 = operator.filter %scan : table<i32, i32> {
    ^bb0 (%orderkey:i32, %shipdate:i32):
        %days_since_epoch_1998_09_02 = arith.constant 10471 : i32
        %2 = arith.cmpi sle, %shipdate, %days_since_epoch_1998_09_02 : i32
        operator.filter.return %2
}
// TODO: rest of the query
// - group by
// - order by