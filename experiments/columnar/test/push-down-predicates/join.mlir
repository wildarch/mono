// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>

#table_A = #columnar.table<"A">
#table_B = #columnar.table<"B">
#column_A_a = #columnar.table_col<#table_A "a" : si64>
#column_B_b = #columnar.table_col<#table_B "b" : si64>

// LHS
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table #column_A_a
    %0 = columnar.read_table #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_table #column_B_b
    %1 = columnar.read_table #column_B_b : <si64>

    // CHECK: %[[#SELECT:]] = columnar.select %[[#A]]
    // CHECK:   columnar.pred %arg0
    // CHECK:   ^bb0(%arg1: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg1, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    //
    // CHECK: %[[#JOIN:]]:2 = columnar.join (%[[#SELECT]]) (%[[#B]])
    %2:2 = columnar.join (%0) (%1) : (!col_si64) (!col_si64)

    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg0 : !col_si64 {
        ^bb0(%arg2: !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %[[#JOIN]]#0, %[[#JOIN]]#1
    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["a", "b"]
}

// RHS
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table #column_A_a
    %0 = columnar.read_table #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_table #column_B_b
    %1 = columnar.read_table #column_B_b : <si64>

    // CHECK: %[[#SELECT:]] = columnar.select %[[#B]]
    // CHECK:   columnar.pred %arg0
    // CHECK:   ^bb0(%arg1: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg1, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    //
    // CHECK: %[[#JOIN:]]:2 = columnar.join (%[[#A]]) (%[[#SELECT]])
    %2:2 = columnar.join (%0) (%1) : (!col_si64) (!col_si64)

    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg1 : !col_si64 {
        ^bb0(%arg2: !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %[[#JOIN]]#0, %[[#JOIN]]#1
    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["a", "b"]
}

// Depends on both sides, no changes
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table #column_A_a
    %0 = columnar.read_table #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_table #column_B_b
    %1 = columnar.read_table #column_B_b : <si64>

    // CHECK: %[[#JOIN:]]:2 = columnar.join (%[[#A]]) (%[[#B]])
    %2:2 = columnar.join (%0) (%1) : (!col_si64) (!col_si64)

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#JOIN]]#0, %[[#JOIN]]#1
    // CHECK:   columnar.pred %arg0, %arg1
    // CHECK:   ^bb0(%arg2: !columnar.col<si64>, %arg3: !columnar.col<si64>):
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg2, %arg3
    // CHECK:     columnar.pred.eval %[[#CMP]]
    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg0, %arg1 : !col_si64, !col_si64 {
        ^bb0(%arg2: !col_si64, %arg3: !col_si64):
            %4 = columnar.cmp EQ %arg2, %arg3 : !col_si64
            columnar.pred.eval %4
        }
    }

    // CHECK: columnar.query.output %[[#SELECT]]#0, %[[#SELECT]]#1
    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["a", "b"]
}

// LHS, with reordered inputs
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table #column_A_a
    %0 = columnar.read_table #column_A_a : <si64>
    // CHECK: %[[#B:]] = columnar.read_table #column_B_b
    %1 = columnar.read_table #column_B_b : <si64>

    // CHECK: %[[#SELECT:]] = columnar.select %[[#A]]
    // CHECK:   columnar.pred %arg0
    // CHECK:   ^bb0(%arg1: !columnar.col<si64>):
    // CHECK:     %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:     %[[#CMP:]] = columnar.cmp EQ %arg1, %[[#CONST]]
    // CHECK:     columnar.pred.eval %[[#CMP]]
    //
    // CHECK: %[[#JOIN:]]:2 = columnar.join (%[[#SELECT]]) (%[[#B]])
    %2:2 = columnar.join (%0) (%1) : (!col_si64) (!col_si64)

    %3:2 = columnar.select %2#1, %2#0 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        columnar.pred %arg1 : !col_si64 {
        ^bb0(%arg2: !col_si64):
            %4 = columnar.constant 42 : si64
            %5 = columnar.cmp EQ %arg2, %4 : !col_si64
            columnar.pred.eval %5
        }
    }

    // CHECK: columnar.query.output %[[#JOIN]]#1, %[[#JOIN]]#0
    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["b", "b"]
}
