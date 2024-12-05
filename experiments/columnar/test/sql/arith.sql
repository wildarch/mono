-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- %0 = columnar.constant 1
-- %1 = columnar.constant 2
-- %2 = columnar.add %0, %1
-- columnar.query.output %2
SELECT 1 + 2;

-- CHECK-LABEL: columnar.query {
-- %0 = columnar.constant 1
-- %1 = columnar.constant 2
-- %2 = columnar.sub %0, %1
-- columnar.query.output %2
SELECT 1 - 2;

-- CHECK-LABEL: columnar.query {
-- %0 = columnar.constant 1
-- %1 = columnar.constant 2
-- %2 = columnar.mul %0, %1
-- columnar.query.output %2
SELECT 1 * 2;

-- CHECK-LABEL: columnar.query {
-- %0 = columnar.constant 1
-- %1 = columnar.constant 2
-- %2 = columnar.div %0, %1
-- columnar.query.output %2
SELECT 1 / 2;