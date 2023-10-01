#pragma once

#include "execution/operator/IR/OperatorOps.h"
#include "mlir/IR/Operation.h"

#include "execution/operator/impl/Operator.h"

namespace execution {

class ProjectOperator : public SingleChildOperator {
private:
  qoperator::ProjectReturnOp _expr;

public:
  inline ProjectOperator(OperatorPtr child, qoperator::ProjectReturnOp expr)
      : SingleChildOperator(OperatorKind::PROJECT, child), _expr(expr) {}

  std::optional<Batch> poll() override;
};

} // namespace execution