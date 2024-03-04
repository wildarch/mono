physicalplan.scan ntuples 1024 at [0xDEADBEEF1, 0xDEADBEEF2] {
^bb0(%0: !physicalplan<block block<i64, i64>> ):
    physicalplan.write_array %0 : block<i64, i64> 
        columns [1] 
        to [0xDEADBEEF3] 
        offptr 0xDEADBEEF4
        capacity 1024
}