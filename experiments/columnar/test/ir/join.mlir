columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si32>
    %1 = columnar.read_lines "/tmp/col1.txt" : <si64>
    %2 = columnar.read_lines "/tmp/col2.txt" : <f32>
    %3 = columnar.read_lines "/tmp/col3.txt" : <f64>
    %4:4 = columnar.join 
        (%0, %1) (%2, %3)
        : (!columnar.col<si32>, !columnar.col<si64>) (!columnar.col<f32>, !columnar.col<f64>)

    columnar.query.output 
        %4#0, %4#1, %4#2, %4#3 
        : !columnar.col<si32>
        , !columnar.col<si64>
        , !columnar.col<f32>
        , !columnar.col<f64>
        ["a", "b", "c", "d"]
}