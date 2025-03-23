// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>

#table_A = #columnar.table<"A" path="/tmp/A.tab">
#table_B = #columnar.table<"B" path="/tmp/B.tab">
#column_A_a = #columnar.table_col<#table_A "a" : si64 path="/tmp/a.col">
#column_B_a = #columnar.table_col<#table_B "a" : si64 path="/tmp/b.col">

columnar.query {
    // CHECK: %[[#A:]] = columnar.read_column #column_A_a
    %0 = columnar.read_column #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_column #column_B_a
    %1 = columnar.read_column #column_B_a : <si64>

    // CHECK: %[[#SELECT_A:]] = columnar.select %[[#A]]
    // CHECK:   columnar.pred %arg0
    // CHECK:   ^bb0(%arg1: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg1, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    //
    // CHECK: %[[#SELECT_B:]] = columnar.select %[[#B]]
    // CHECK:   columnar.pred %arg0
    // CHECK:   ^bb0(%arg1: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg1, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    //
    // CHECK: %[[#UNION:]] = columnar.union (%[[#SELECT_A]]) (%[[#SELECT_B]])
    %2 = columnar.union (%0) (%1) : !col_si64

    %3 = columnar.select %2 : !col_si64 {
    ^bb0(%arg0: !col_si64):
        columnar.pred %arg0 : !col_si64 {
        ^bb0(%arg1: !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg1, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %[[#UNION]]
    columnar.query.output %3 : !col_si64 ["a"]
}