#include "execution/expression/ExpressionEvaluator.h"
#include "execution/Batch.h"
#include "execution/expression/BinaryOperatorExpression.h"
#include "execution/expression/ColumnExpression.h"
#include "execution/expression/ConstantExpression.h"
#include "execution/expression/ExpressionVisitor.h"

namespace execution {

ConstantValue applyBinOp(ConstantValue lhs, BinaryOperator op,
                         ConstantValue rhs) {
  switch (op) {
  case BinaryOperator::LE_INT32:
    return std::get<int32_t>(lhs) <= std::get<int32_t>(rhs);
  }

  throw std::logic_error("invalid enum");
}

ConstantValue evaluate(const Expression &expr, const Batch &batch,
                       uint32_t row) {
  return ExpressionVisitor{
      [&](const BinaryOperatorExpression &expr) {
        return applyBinOp(evaluate(*expr.lhs(), batch, row), expr.op(),
                          evaluate(*expr.rhs(), batch, row));
      },
      [&](const ColumnExpression &expr) -> ConstantValue {
        switch (expr.type()) {
#define CASE(t)                                                                \
  case PhysicalColumnType::t:                                                  \
    return batch.columns()                                                     \
        .at(expr.idx().value())                                                \
        .get<PhysicalColumnType::t>()[row];                                    \
    break;
          CASE(INT32)
          CASE(DOUBLE)
          CASE(STRING_PTR)
#undef CASE
        }
        throw std::logic_error("invalid enum");
      },
      [](const ConstantExpression &expr) { return expr.value(); },
  }
      .visit(expr);
}

} // namespace execution