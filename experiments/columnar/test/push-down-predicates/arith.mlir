// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>
!col_i1 = !columnar.col<i1>

#table_A = #columnar.table<"A">
#column_A_a = #columnar.table_col<#table_A "a" : !columnar.dec>
#column_A_b = #columnar.table_col<#table_A "b" : !columnar.dec>

// Not part of predicate
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_column #column_A_a
    %0 = columnar.read_column #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_column #column_A_b
    %1 = columnar.read_column #column_A_b : <si64>

    %2 = columnar.cmp EQ %0, %1 : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#A]], %[[#B]]
    // CHECK:   columnar.pred %arg0
    // CHECK:   ^bb0(%arg2: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg2, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    %3:2 = columnar.select %0, %2 : !col_si64, !col_i1 {
    ^bb0(%arg0: !col_si64, %arg1: !col_i1):
        columnar.pred %arg0 : !col_si64 {
        ^bb0(%arg2 : !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: %[[#CMP:]] = columnar.cmp EQ %[[#SELECT]]#0, %[[#SELECT]]#1
    //
    // CHECK: columnar.query.output %[[#SELECT]]#0, %[[#CMP]]
    columnar.query.output %3#0, %3#1 : !col_si64, !col_i1 ["a", "a=b"]
}

// Only needed for predicate
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_column #column_A_a
    %0 = columnar.read_column #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_column #column_A_b
    %1 = columnar.read_column #column_A_b : <si64>

    %2 = columnar.cmp EQ %0, %1 : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#A]], %[[#B]]
    // CHECK:   columnar.pred %arg0, %arg1
    // CHECK:   ^bb0(%arg2: !columnar.col<si64>, %arg3: !columnar.col<si64>):
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg2, %arg3
    // CHECK:     columnar.pred.eval %[[#CMP]]
    %3:2 = columnar.select %0, %2 : !col_si64, !col_i1 {
    ^bb0(%arg0: !col_si64, %arg1: !col_i1):
        columnar.pred %arg1 : !col_i1 {
        ^bb0(%arg2 : !col_i1):
            columnar.pred.eval %arg2
        }
    }

    // CHECK: columnar.query.output %[[#SELECT]]#0
    columnar.query.output %3#0 : !col_si64 ["a"]
}

// Needed for predicate and output
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_column #column_A_a
    %0 = columnar.read_column #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_column #column_A_b
    %1 = columnar.read_column #column_A_b : <si64>

    %2 = columnar.cmp EQ %0, %1 : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#A]], %[[#B]]
    // CHECK:   columnar.pred %arg0, %arg1
    // CHECK:   ^bb0(%arg2: !columnar.col<si64>, %arg3: !columnar.col<si64>):
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg2, %arg3
    // CHECK:     columnar.pred.eval %[[#CMP]]
    %3:2 = columnar.select %0, %2 : !col_si64, !col_i1 {
    ^bb0(%arg0: !col_si64, %arg1: !col_i1):
        columnar.pred %arg1 : !col_i1 {
        ^bb0(%arg2 : !col_i1):
            columnar.pred.eval %arg2
        }
    }

    // CHECK: %[[#CMP:]] = columnar.cmp EQ %[[#SELECT]]#0, %[[#SELECT]]#1
    //
    // CHECK: columnar.query.output %[[#SELECT]]#0, %[[#CMP]]
    columnar.query.output %3#0, %3#1 : !col_si64, !col_i1 ["a", "a=b"]
}