%scan = operator.scan_parquet 
    "/home/daan/workspace/duckdb/duckdb_benchmark_data/tpch_sf1_parquet/lineitem.parquet" 
    [
        "l_orderkey", 
        "l_shipdate",
        "l_returnflag",
        "l_linestatus"] : table<i32, i32, !operator.varchar, !operator.varchar>
%1 = operator.filter %scan : table<i32, i32, !operator.varchar, !operator.varchar> {
    ^bb0 (%orderkey:i32, 
          %shipdate:i32,
          %returnflag:!operator.varchar,
          %linestatus:!operator.varchar):
        %days_since_epoch_1998_09_02 = arith.constant 10471 : i32
        %2 = arith.cmpi sle, %shipdate, %days_since_epoch_1998_09_02 : i32
        operator.filter.return %2
}
%2 = operator.aggregate %1 : table<i32, i32, !operator.varchar, !operator.varchar> -> table<!operator.varchar, !operator.varchar> {
    ^bb0 (%orderkey:i32, 
          %shipdate:i32,
          %returnflag:!operator.varchar,
          %linestatus:!operator.varchar):
    %key_returnflag = operator.aggregate.key %returnflag : !operator.varchar -> aggregator<!operator.varchar>
    %key_linestatus = operator.aggregate.key %linestatus : !operator.varchar -> aggregator<!operator.varchar>
    operator.aggregate.return (%key_returnflag, %key_linestatus : !operator<aggregator aggregator<!operator.varchar>>, !operator<aggregator aggregator<!operator.varchar>>)
}
// TODO: rest of the query
// - group by: add non-key columns
// - order by