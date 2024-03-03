physicalplan.pipeline {
^bb0:
    %0 = physicalplan.scan 1024 tuples at [0xDEADBEEF1, 0xDEADBEEF2] : block<i64, i64>
    %1 = physicalplan.compute %0 : block<i64, i64> -> block<i64, i64, i1> {
    ^bb0(%a:i64, %b:i64):
        %c5 = arith.constant 5 : i64
        %c = arith.cmpi slt, %a, %c5 : i64
        physicalplan.compute.return %c : i1
    } 
    %2 = physicalplan.filter %1 : block<i64, i64, i1> 2
    %3 = physicalplan.compute %2 : block<i64, i64, i1> -> block<i64, i64, i1, i64> {
    ^bb0(%a:i64, %b:i64, %c:i1):
        %d = arith.addi %a, %b : i64
        physicalplan.compute.return %d : i64
    }
    physicalplan.write_array %3 : block<i64, i64, i1, i64> 
        columns [3] 
        to [0xDEADBEEF3] 
        offptr 0xDEADBEEF4
        capacity 1024
}