physicalplan.scan ntuples 1024 at [0xDEADBEEF1, 0xDEADBEEF2] {
^bb0(%0: !physicalplan<block block<i64, i64>> ):
    %3 = physicalplan.compute %0 : block<i64, i64> -> block<i64, i64, i64> {
    ^bb0(%a:i64, %b:i64, %c:i1):
        %d = arith.addi %a, %b : i64
        physicalplan.compute.return %d : i64
    }
    physicalplan.write_array %3 : block<i64, i64, i64> 
        columns [2] 
        to [0xDEADBEEF3] 
        offptr 0xDEADBEEF4
        capacity 1024
}