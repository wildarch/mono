%0 = operator.scan_parquet "/home/daan/workspace/duckdb/duckdb_benchmark_data/tpch_sf1_parquet/lineitem.parquet" ["l_orderkey"] : table<i32>
%1 = operator.filter %0 : table<i32> {
    ^bb0 (%arg0:i64):
        %2 = arith.constant 1 : i1
        operator.filter.return %2
}