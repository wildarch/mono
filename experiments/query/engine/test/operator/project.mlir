%0 = operator.scan_parquet "/home/daan/workspace/duckdb/duckdb_benchmark_data/tpch_sf1_parquet/lineitem.parquet" ["l_orderkey"] : table<i32>
%1 = operator.project %0 : table<i32> -> table<i32> {
    ^bb0 (%orderkey:i32):
        %one = arith.constant 1 : i32
        %3 = arith.addi %orderkey, %one : i32
        operator.project.return(%3:i32)
}