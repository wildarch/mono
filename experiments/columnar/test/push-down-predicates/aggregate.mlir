!col_si64 = !columnar.col<si64>
!col_f64 = !columnar.col<f64>

// Basic
columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>

    %2:2 = columnar.aggregate 
        group %0 : !col_si64
        aggregate %1 : !col_si64 [SUM]

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

// With COUNT, changing the type of the second block arg
columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <f64>

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

// Depends on aggregation value
columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>

    %2:2 = columnar.aggregate 
        group %0 : !col_si64
        aggregate %1 : !col_si64 [SUM]

    %3:2 = columnar.select %2#0, %2#1 : !col_si64, !col_si64 {
    ^bb0(%arg0: !col_si64, %arg1: !col_si64):
        %4 = columnar.constant <42 : si64> : !col_si64
        %5 = columnar.cmp EQ %arg1, %4 : !col_si64
        columnar.select.return %5
    }

    columnar.query.output 
        %3#0, %3#1
        : !col_si64
        , !col_si64
        ["key", "sum"]
}