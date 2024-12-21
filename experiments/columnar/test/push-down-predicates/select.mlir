// RUN: mlir-opt --push-down-predicates %s | FileCheck %s
!col_si64 = !columnar.col<si64>

// CHECK-LABEL: columnar.query {
columnar.query {
  // CHECK: %[[#A:]] = columnar.read_table "A"
  %0 = columnar.read_table "A" "a" : <si64>

  // CHECK %[[#SELECT:]] = columnar.select %[[#A]]
  %1 = columnar.select %0 : !col_si64 {
  ^bb0(%arg0: !col_si64):
    // CHECK: columnar.pred %arg0
    // CHECK:   %[[#CONST:]] = columnar.constant 42
    // CHECK:   %[[#CMP:]] = columnar.cmp LT %[[#CONST]], %arg1
    // CHECK:   columnar.pred.eval %[[#CMP]]
    //
    // CHECK: columnar.pred %arg0
    // CHECK:   %[[#CONST:]] = columnar.constant 100
    // CHECK:   %[[#CMP:]] = columnar.cmp LT %arg1, %[[#CONST]]
    // CHECK:   columnar.pred.eval %[[#CMP]]
    columnar.pred %arg0 : !col_si64 {
    ^bb0(%arg1: !col_si64):
      %2 = columnar.constant 42 : si64
      %3 = columnar.cmp LT %2, %arg1 : !col_si64
      columnar.pred.eval %3
    }
  }

  %2 = columnar.select %1 : !col_si64 {
  ^bb0(%arg0: !col_si64):
    columnar.pred %arg0 : !col_si64 {
    ^bb0(%arg1: !col_si64):
      %2 = columnar.constant 100 : si64
      %3 = columnar.cmp LT %arg1, %2 : !col_si64
      columnar.pred.eval %3
    }
  }

  columnar.query.output %2 : !col_si64 ["a"]
}