#pragma once

#include "execution/expression/Expression.h"
#include "execution/operator/Operator.h"

namespace execution {

class FilterOperator : public SingleChildOperator {
private:
  ExpressionPtr _expr;

public:
  inline FilterOperator(OperatorPtr child, ExpressionPtr expr)
      : SingleChildOperator(OperatorKind::PARQUET_SCAN, child), _expr(expr) {}

  std::optional<Batch> poll() override;
};

} // namespace execution