columnar.query {
    // From Employee
    %id = columnar.read_table "Employee" "id" : <ui64>
    %name = columnar.read_table "Employee" "name" : <!columnar.str>

    // From TimeTransaction
    %employee_id = columnar.read_table "TimeTransaction" "employee_id" : <ui64>
    %hours = columnar.read_table "TimeTransaction" "hours" : <f64>

    // Join all from clauses
    %join:4 = columnar.join (%id, %name) (%employee_id, %hours) 
        : (!columnar.col<ui64>, !columnar.col<!columnar.str>)
        (!columnar.col<ui64>, !columnar.col<f64>)

    // Apply WHERE clauses
    %select:2 = columnar.select
        %join#1, %join#3 : !columnar.col<!columnar.str>, !columnar.col<f64>
        predicate %join#0, %join#2 : !columnar.col<ui64>, !columnar.col<ui64> {
    ^bb0(%arg_id : !columnar.col<ui64>, %arg_employee_id : !columnar.col<ui64>):
        %eq = columnar.cmp EQ %arg_id, %arg_employee_id : <ui64>
        columnar.select.return %eq
    }

    // Projections before aggregation (N/A)

    // Group By & Aggregate
    %aggregate:2 = columnar.aggregate
        group %select#0 : !columnar.col<!columnar.str>
        aggregate %select#1 : !columnar.col<f64> [SUM]

    // Projections after aggregation (N/A)
    columnar.query.output 
        %aggregate#0, %aggregate#1 : !columnar.col<!columnar.str>, !columnar.col<f64>
        ["name", "hours"]
}