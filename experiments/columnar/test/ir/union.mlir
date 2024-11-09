columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <f64>
    %2 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %3 = columnar.read_lines "/tmp/col1.txt" : <f64>

    %4:2 = columnar.union (%0, %1) (%2, %3) : !columnar.col<si64>, !columnar.col<f64>

    columnar.query.output %4#0, %4#1 : !columnar.col<si64>, !columnar.col<f64> ["a", "b"]
}