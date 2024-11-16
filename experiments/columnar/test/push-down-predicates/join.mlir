// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>

columnar.query {
    %0 = columnar.read_table "A" "a" : <si64>
    %1 = columnar.read_table "B" "b" : <si64>

    %2:2 = columnar.join (%0) (%1) : (!col_si64) (!col_si64)

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
        ["a", "b"]
}