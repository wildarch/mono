// RUN columnar-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>
!col_f64 = !columnar.col<f64>

#table_A = #columnar.table<"A" path="/tmp/A.tab">
#column_A_a = #columnar.table_col<#table_A "a" : si64 path="/tmp/a.col">
#column_A_b = #columnar.table_col<#table_A "b" : si64 path="/tmp/b.col">
#column_A_c = #columnar.table_col<#table_A "c" : f64 path="/tmp/c.col">

// Basic
// CHECK-LABEL: columnar.query {
columnar.query {
    %0 = columnar.read_column #column_A_a : <si64>
    %1 = columnar.read_column #column_A_b : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %0, %1
    // CHECK:   columnar.pred %arg0 : !columnar.col<si64> {
    // CHECK:   ^bb0(%arg2: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg2, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    // CHECK:   }
    //
    // CHECK: %[[#AGG:]]:2 = columnar.aggregate group %[[#SELECT]]#0
    // CHECK-SAME: aggregate %[[#SELECT]]#1
    %2:2 = columnar.aggregate
        group %0 : !col_si64
        aggregate %1 : !col_si64 [SUM]

    // Note: removed because no predicates left.
    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg0 : !col_si64 {
        ^bb0(%arg2 : !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %[[#AGG]]#0, %[[#AGG]]#1
    columnar.query.output
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}

// With COUNT, changing the type of the second block arg
// CHECK-LABEL: columnar.query {
columnar.query {
    %0 = columnar.read_column #column_A_a : <si64>
    %1 = columnar.read_column #column_A_c : <f64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %0, %1 : !columnar.col<si64>, !columnar.col<f64>
    // CHECK: ^bb0(%arg0: !columnar.col<si64>, %arg1: !columnar.col<f64>):
    //
    // CHECK: columnar.aggregate
    // CHECK-SAME: group %[[#SELECT]]#0
    // CHECK-SAME: aggregate %[[#SELECT]]#1
    %2:2 = columnar.aggregate
        group %0 : !col_si64
        aggregate %1 : !col_f64 [COUNT]

    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg0 : !col_si64 {
        ^bb0(%arg2 : !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    columnar.query.output
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}

// Depends on aggregation value, no changes
// CHECK-LABEL: columnar.query {
columnar.query {
    %0 = columnar.read_column #column_A_a : <si64>
    %1 = columnar.read_column #column_A_b : <si64>

    // CHECK: %2:2 = columnar.aggregate
    // CHECK-SAME: group %0
    // CHECK-SAME: aggregate %1
    %2:2 = columnar.aggregate
        group %0 : !col_si64
        aggregate %1 : !col_si64 [SUM]

    // CHECK: %3:2 = columnar.select %2#0, %2#1
    // CHECK:  columnar.pred %arg1 : !columnar.col<si64> {
    // CHECK:  ^bb0(%arg2: !columnar.col<si64>):
    // CHECK:    %[[#CONST:]] = columnar.constant 42
    // CHECK:    %[[#CMP:]] = columnar.cmp EQ %arg2, %[[#CONST]]
    // CHECK:    columnar.pred.eval %[[#CMP]]
    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg1 : !col_si64 {
        ^bb0(%arg2: !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %3#0, %3#1
    columnar.query.output
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}

// With swapped column order
// CHECK-LABEL: columnar.query {
columnar.query {
    %0 = columnar.read_column #column_A_a : <si64>
    %1 = columnar.read_column #column_A_b : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %0, %1
    // CHECK:   columnar.pred %arg1
    //
    // CHECK: %[[#AGG:]]:2 = columnar.aggregate group %[[#SELECT]]#0, %[[#SELECT]]#1
    %2:2 = columnar.aggregate
        group %0, %1 : !col_si64, !col_si64 []

    %3:2 = columnar.select %2#1, %2#0 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg0 : !col_si64 {
            ^bb0(%arg2: !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %[[#AGG]]#0, %[[#AGG]]#1
    columnar.query.output
        %3#1, %3#0
        : !col_si64
        , !col_si64
        ["key", "sum"]
}
