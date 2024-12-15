-- RUN: translate --import-sql %s | FileCheck %s

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.constant #columnar<str "Hello, World!">
-- CHECK: %1 = columnar.constant #columnar<str "%World!">
-- CHECK: %2 = columnar.like %0 like %1
-- CHECK: columnar.query.output %2 : !columnar.col<i1> ["match"]
SELECT 'Hello, World!' LIKE '%World!' AS match;

-- CHECK-LABEL: columnar.query {
-- CHECK: %0 = columnar.constant #columnar<str "Hello, World!">
-- CHECK: %1 = columnar.constant #columnar<str "%World!">
-- CHECK: %2 = columnar.like %0 like %1
-- CHECK: %3 = columnar.not %2
-- CHECK: columnar.query.output %3 : !columnar.col<i1> ["match"]
SELECT 'Hello, World!' NOT LIKE '%World!' AS match;