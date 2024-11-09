columnar.query {
    %0 = columnar.read_lines "/tmp/data.txt" : <si64>
    columnar.query.output %0 : !columnar.col<si64> ["data"]
}