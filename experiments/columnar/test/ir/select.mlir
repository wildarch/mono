columnar.query {
    %0 = columnar.read_lines "/tmp/col0.txt" : <si64>
    %1 = columnar.read_lines "/tmp/col1.txt" : <i1>
    %2:2 = columnar.select %0, %1 : !columnar.col<si64>, !columnar.col<i1> {
    ^bb0(%arg0 : !columnar.col<si64>, %arg1 : !columnar.col<i1>):
        columnar.select.return %arg1
    }

    columnar.query.output %2#0 : !columnar.col<si64>
        ["value"]
}