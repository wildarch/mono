// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>
!col_i1 = !columnar.col<i1>

// Not part of predicate
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table "A" "a"
    %0 = columnar.read_table "A" "a" : <si64>
    // CHECK: %[[#B:]] = columnar.read_table "A" "b"
    %1 = columnar.read_table "A" "b" : <si64>

    %2 = columnar.cmp EQ %0, %1 : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#A]], %[[#B]]
    // CHECK:   %[[#CONST:]] = columnar.constant<42 : si64>
    // CHECK:   %[[#CMP:]] = columnar.cmp EQ %arg0, %[[#CONST]]
    // CHECK:   columnar.select.return %[[#CMP]]
    %3:2 = columnar.select %0, %2 : !col_si64, !col_i1 {
    ^bb0(%arg0: !col_si64, %arg1: !col_i1):
        %4 = columnar.constant <42 : si64> : !col_si64
        %5 = columnar.cmp EQ %arg0, %4 : !col_si64
        columnar.select.return %5
    }

    // CHECK: %[[#CMP:]] = columnar.cmp EQ %[[#SELECT]]#0, %[[#SELECT]]#1
    //
    // CHECK: columnar.query.output %[[#SELECT]]#0, %[[#CMP]]
    columnar.query.output %3#0, %3#1 : !col_si64, !col_i1 ["a", "a=b"]
}

// Only needed for predicate
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table "A" "a"
    %0 = columnar.read_table "A" "a" : <si64>
    // CHECK: %[[#B:]] = columnar.read_table "A" "b"
    %1 = columnar.read_table "A" "b" : <si64>

    %2 = columnar.cmp EQ %0, %1 : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#A]], %[[#B]]
    // CHECK:   %[[#CMP:]] = columnar.cmp EQ %arg0, %arg1
    // CHECK:   columnar.select.return %[[#CMP]]
    %3:2 = columnar.select %0, %2 : !col_si64, !col_i1 {
    ^bb0(%arg0: !col_si64, %arg1: !col_i1):
        columnar.select.return %arg1
    }

    // CHECK: columnar.query.output %[[#SELECT]]#0
    columnar.query.output %3#0 : !col_si64 ["a"]
}

// Needed for predicate and output
// CHECK-LABEL: columnar.query {
columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table "A" "a"
    %0 = columnar.read_table "A" "a" : <si64>
    // CHECK: %[[#B:]] = columnar.read_table "A" "b"
    %1 = columnar.read_table "A" "b" : <si64>

    %2 = columnar.cmp EQ %0, %1 : <si64>

    // CHECK: %[[#SELECT:]]:2 = columnar.select %[[#A]], %[[#B]]
    // CHECK:   %[[#CMP:]] = columnar.cmp EQ %arg0, %arg1
    // CHECK:   columnar.select.return %[[#CMP]]
    %3:2 = columnar.select %0, %2 : !col_si64, !col_i1 {
    ^bb0(%arg0: !col_si64, %arg1: !col_i1):
        columnar.select.return %arg1
    }

    // CHECK: %[[#CMP:]] = columnar.cmp EQ %[[#SELECT]]#0, %[[#SELECT]]#1
    //
    // CHECK: columnar.query.output %[[#SELECT]]#0, %[[#CMP]]
    columnar.query.output %3#0, %3#1 : !col_si64, !col_i1 ["a", "a=b"]
}