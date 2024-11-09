!col_si64 = !columnar.col<si64>
!col_f64 = !columnar.col<f64>

columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>
    %2 = columnar.read_lines "/tmp/col2.txt" : <si64>
    %3 = columnar.read_lines "/tmp/col3.txt" : <f64>
    %4:4 = columnar.aggregate 
        group %0, %1 : !col_si64, !col_si64
        aggregate %2, %3 : !col_si64, !col_f64 [SUM, COUNT]
    columnar.query.output 
        %4#0, %4#1, %4#2, %4#3 
        : !col_si64
        , !col_si64
        , !col_si64
        , !col_si64
        ["key0", "key1", "sum", "count"]
}