#pragma once

#include "execution/Batch.h"
#include "execution/operator/IR/OperatorOps.h"
#include "mlir/IR/Operation.h"

#include "execution/operator/impl/Operator.h"

namespace execution {

class ProjectOperator : public SingleChildOperator {
private:
  qoperator::ProjectReturnOp _expr;
  std::vector<PhysicalColumnType> _outputColumnTypes;

public:
  ProjectOperator(OperatorPtr child, qoperator::ProjectReturnOp expr);

  std::optional<Batch> poll() override;
};

} // namespace execution