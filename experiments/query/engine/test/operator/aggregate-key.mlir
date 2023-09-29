%0 = operator.scan_parquet "/home/daan/workspace/duckdb/duckdb_benchmark_data/tpch_sf1_parquet/lineitem.parquet" ["l_orderkey", "l_shipdate"] : table<i32, i32>
%1 = operator.aggregate %0 : table<i32, i32> -> table<i32, i32> {
    ^bb0 (%arg0:i32, %arg1:i32):
        %2 = operator.aggregate.key %arg0 : i32 -> aggregator<i32>
        %3 = operator.aggregate.key %arg1 : i32 -> aggregator<i32>
        operator.aggregate.return (%3, %3 : !operator<aggregator aggregator<i32>>, !operator<aggregator aggregator<i32>>)
}