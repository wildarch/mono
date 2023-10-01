#pragma once

#include "execution/expression/ConstantExpression.h"
#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/impl/Operator.h"
namespace execution {

class AggregateOperator : public SingleChildOperator {
private:
  qoperator::AggregateReturnOp _retOp;

  struct Hasher {
    size_t operator()(const std::vector<ConstantValue> &keys) const;
  };
  std::unordered_map<std::vector<ConstantValue>, std::vector<ConstantValue>,
                     Hasher>
      _aggregate;

public:
  inline AggregateOperator(OperatorPtr child,
                           qoperator::AggregateReturnOp retOp)
      : SingleChildOperator(OperatorKind::AGGREGATE, child), _retOp(retOp) {}

  std::optional<Batch> poll() override;
};

} // namespace execution