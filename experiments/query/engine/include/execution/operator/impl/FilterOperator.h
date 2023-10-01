#pragma once

#include "execution/operator/IR/OperatorOps.h"
#include "mlir/IR/Operation.h"

#include "execution/expression/Expression.h"
#include "execution/operator/impl/Operator.h"

namespace execution {

class FilterOperator : public SingleChildOperator {
private:
  qoperator::FilterReturnOp _expr;

public:
  inline FilterOperator(OperatorPtr child, qoperator::FilterReturnOp expr)
      : SingleChildOperator(OperatorKind::FILTER, child), _expr(expr) {}

  std::optional<Batch> poll() override;
};

} // namespace execution