// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>

columnar.query {
    // CHECK: %[[#A:]] = columnar.read_table "A"
    %0 = columnar.read_table "A" "a" : <si64>
    // CHECK: %[[#B:]] = columnar.read_table "B"
    %1 = columnar.read_table "B" "a" : <si64>

    // CHECK: %[[#SELECT_A:]] = columnar.select %[[#A]]
    // CHECK:   %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:   %[[#CMP:]] = columnar.cmp EQ %arg0, %[[#CONST]]
    // CHECK:   columnar.select.return %[[#CMP]]
    //
    // CHECK: %[[#SELECT_B:]] = columnar.select %[[#B]]
    // CHECK:   %[[#CONST:]] = columnar.constant 42 : si64
    // CHECK:   %[[#CMP:]] = columnar.cmp EQ %arg0, %[[#CONST]]
    // CHECK:   columnar.select.return %[[#CMP]]
    //
    // CHECK: %[[#UNION:]] = columnar.union (%[[#SELECT_A]]) (%[[#SELECT_B]])
    %2 = columnar.union (%0) (%1) : !col_si64

    %3 = columnar.select %2 : !col_si64 {
    ^bb0(%arg0: !col_si64):
        %4 = columnar.constant 42 : si64
        %5 = columnar.cmp EQ %arg0, %4 : !col_si64
        columnar.select.return %5
    }

    // CHECK: columnar.query.output %[[#UNION]]
    columnar.query.output %3 : !col_si64 ["a"]
}