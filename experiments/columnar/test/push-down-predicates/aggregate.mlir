// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>
!col_f64 = !columnar.col<f64>

// Basic
columnar.query {
    // CHECK: %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    // CHECK: %1 = columnar.read_lines "/tmp/col1.txt" : <si64>
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %0, %1
    // CHECK:   %[[#CONST:]] = columnar.constant<42 : si64> : <si64>
    // CHECK:   %[[#CMP:]] = columnar.cmp EQ %arg0, %[[#CONST]] : <si64>
    // CHECK:   columnar.select.return %[[#CMP]]
    //
    // CHECK: %[[#AGG:]]:2 = columnar.aggregate group %[[#SELECT]]#0
    // CHECK-SAME: aggregate %[[#SELECT]]#1
    %2:2 = columnar.aggregate 
        group %0 : !col_si64
        aggregate %1 : !col_si64 [SUM]

    // Note: removed because no predicates left.
    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        %4 = columnar.constant <42 : si64> : !col_si64
        %5 = columnar.cmp EQ %arg0, %4 : !col_si64
        columnar.select.return %5
    }

    // CHECK: columnar.query.output %[[#AGG]]#0, %[[#AGG]]#1
    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}

// With COUNT, changing the type of the second block arg
columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <f64>

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
        %4 = columnar.constant <42 : si64> : !col_si64
        %5 = columnar.cmp EQ %arg0, %4 : !col_si64
        columnar.select.return %5
    }

    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}

// Depends on aggregation value, no changes
columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>

    // CHECK: %2:2 = columnar.aggregate 
    // CHECK-SAME: group %0
    // CHECK-SAME: aggregate %1
    %2:2 = columnar.aggregate 
        group %0 : !col_si64
        aggregate %1 : !col_si64 [SUM]

    // CHECK: %3:2 = columnar.select %2#0, %2#1
    // CHECK:   %4 = columnar.constant<42 : si64> : <si64>
    // CHECK:   %5 = columnar.cmp EQ %arg1, %4 : <si64>
    // CHECK:   columnar.select.return %5
    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        %4 = columnar.constant <42 : si64> : !col_si64
        %5 = columnar.cmp EQ %arg1, %4 : !col_si64
        columnar.select.return %5
    }

    // CHECK: columnar.query.output %3#0, %3#1
    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}

// With swapped column order
columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %0, %1
    // CHECK:   columnar.cmp EQ %arg1, %4 : <si64>
    //
    // CHECK: %[[#AGG:]]:2 = columnar.aggregate group %[[#SELECT]]#0, %[[#SELECT]]#1
    %2:2 = columnar.aggregate 
        group %0, %1 : !col_si64, !col_si64 []

    %3:2 = columnar.select %2#1, %2#0 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        %4 = columnar.constant <42 : si64> : !col_si64
        %5 = columnar.cmp EQ %arg0, %4 : !col_si64
        columnar.select.return %5
    }

    // CHECK: columnar.query.output %[[#AGG]]#0, %[[#AGG]]#1
    columnar.query.output 
        %3#1, %3#0
        : !col_si64
        , !col_si64
        ["key", "sum"]
}