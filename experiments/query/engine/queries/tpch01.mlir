%scan = operator.scan_parquet 
    "/home/daan/workspace/duckdb/duckdb_benchmark_data/tpch_sf1_parquet/lineitem.parquet" 
    [
        "l_shipdate",
        "l_returnflag",
        "l_linestatus",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax"] : table<
            i32,                    // l_shipdate (DATE)
            !operator.varchar,      // l_returnflag (VARCHAR)
            !operator.varchar,      // l_linestatus (VARCHAR)
            i64,                    // l_quantity (DECIMAL(15,2))
            i64,                    // l_extendedprice (DECIMAL(15,2))
            i64,                    // l_discount (DECIMAL(15,2))
            i64>                    // l_tax (DECIMAL(15,2))
%1 = operator.filter %scan : table<i32, !operator.varchar, !operator.varchar, i64, i64, i64, i64> {
    ^bb0 (
        %shipdate:i32,
        %returnflag:!operator.varchar,
        %linestatus:!operator.varchar,
        %quantity:i64,
        %extendedprice:i64,
        %discount:i64,
        %tax:i64):
    %days_since_epoch_1998_09_02 = arith.constant 10471 : i32
    %2 = arith.cmpi sle, %shipdate, %days_since_epoch_1998_09_02 : i32
    operator.filter.return %2
}
%2 = operator.aggregate %1 : table<i32, !operator.varchar, !operator.varchar, i64, i64, i64, i64> -> table<
        !operator.varchar,  // l_returnflag
        !operator.varchar,  // l_linestatus
        i64,                // sum_qty
        i64,                // sum_base_price
        i64,                // sum_disc_price
        i64,                // sum_charge
        i64,                // sum_discount
        i64                 // count_order
        > {
    ^bb0 (
        %shipdate:i32,
        %returnflag:!operator.varchar,
        %linestatus:!operator.varchar,
        %quantity:i64,
        %extendedprice:i64,
        %discount:i64,
        %tax:i64):
    %key_returnflag = operator.aggregate.key %returnflag : !operator.varchar -> aggregator<!operator.varchar>
    %key_linestatus = operator.aggregate.key %linestatus : !operator.varchar -> aggregator<!operator.varchar>
    %sum_qty = operator.aggregate.sum %quantity : i64 -> aggregator<i64>
    %sum_base_price = operator.aggregate.sum %extendedprice : i64 -> aggregator<i64>

    // at decimal scale -2 we get 1 => 1*10^2 = 100
    %c100 = arith.constant 100 : i64
    %one = arith.constant 1 : i64
    %0 = arith.subi %c100, %discount : i64
    %disc_price = arith.muli %extendedprice, %0 : i64
    %sum_disc_price = operator.aggregate.sum %disc_price : i64 -> aggregator<i64>

    %2 = arith.addi %c100, %tax : i64
    %3 = arith.muli %disc_price, %2 : i64
    %sum_charge = operator.aggregate.sum %3 : i64 -> aggregator<i64>

    %sum_discount = operator.aggregate.sum %discount : i64 -> aggregator<i64>

    %count_order = operator.aggregate.count aggregator<i64>

    operator.aggregate.return (
        %key_returnflag, 
        %key_linestatus,
        %sum_qty,
        %sum_base_price,
        %sum_disc_price,
        %sum_charge,
        %sum_discount,
        %count_order: 
            !operator<aggregator aggregator<!operator.varchar>>,    // returnflag
            !operator<aggregator aggregator<!operator.varchar>>,    // linestatus
            !operator<aggregator aggregator<i64>>,                  // sum_qty
            !operator<aggregator aggregator<i64>>,                  // sum_base_price
            !operator<aggregator aggregator<i64>>,                  // sum_disc_price
            !operator<aggregator aggregator<i64>>,                  // sum_charge
            !operator<aggregator aggregator<i64>>,                  // sum_discount
            !operator<aggregator aggregator<i64>>)                  // count_order
}
%3 = operator.project %2 : table<!operator.varchar, !operator.varchar, i64, i64, i64, i64, i64, i64> -> table<
        !operator.varchar,  // l_returnflag
        !operator.varchar,  // l_linestatus
        i64,                // sum_qty
        i64,                // sum_base_price
        i64,                // sum_disc_price
        i64,                // sum_charge
        i64,                // avg_qty
        i64,                // avg_price
        i64,                // avg_disc
        i64                 // count_order
        > {
    ^bb0 (
        %returnflag:!operator.varchar,
        %linestatus:!operator.varchar,
        %sum_qty:i64,
        %sum_base_price:i64,
        %sum_disc_price:i64,
        %sum_charge:i64,
        %sum_discount:i64,
        %count_order:i64):
    %avg_qty = arith.divsi %sum_qty, %count_order : i64                         // TODO: decimal
    %avg_price = arith.divsi %sum_base_price, %count_order : i64                // TODO: decimal
    %avg_disc = arith.divsi %sum_discount, %count_order : i64                   // TODO: decimal
    operator.project.return (
        %returnflag, 
        %linestatus,
        %sum_qty,
        %sum_base_price,
        %sum_disc_price,
        %sum_charge,
        %avg_qty,
        %avg_price,
        %avg_disc,
        %count_order: 
            !operator.varchar,    // returnflag
            !operator.varchar,    // linestatus
            i64,                  // sum_qty
            i64,                  // sum_base_price
            i64,                  // sum_disc_price
            i64,                  // sum_charge
            i64,                  // avg_qty
            i64,                  // avg_price
            i64,                  // avg_disc
            i64)                  // count_order
}


// TODO: rest of the query
// - order by