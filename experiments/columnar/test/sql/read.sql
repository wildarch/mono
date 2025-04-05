-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK: columnar.query {
-- CHECK:   %0 = columnar.read_column #column_nation_n_nationkey : <i64>
-- CHECK:   %1 = columnar.read_column #column_nation_n_name : <!columnar.str>
-- CHECK:   %2 = columnar.read_column #column_nation_n_regionkey : <i64>
-- CHECK:   %3 = columnar.read_column #column_nation_n_comment : <!columnar.str>
-- CHECK:   columnar.query.output %0 : !columnar.col<i64> ["n_nationkey"]
-- CHECK: }
SELECT n_nationkey
FROM nation;