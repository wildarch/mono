#include "execution/operator/impl/ProjectOperator.h"
#include "execution/Batch.h"
#include "execution/expression/ExpressionEvaluator.h"
#include "execution/operator/impl/Operator.h"

namespace execution {

ProjectOperator::ProjectOperator(OperatorPtr child,
                                 qoperator::ProjectReturnOp expr)
    : SingleChildOperator(OperatorKind::PROJECT, child), _expr(expr) {
  for (const auto &val : expr->getOperands()) {
    _outputColumnTypes.emplace_back(mlirToPhysicalType(val.getType()));
  }
}

template <PhysicalColumnType type>
static void projectColumn(const Batch &input, Batch &output, size_t col,
                          mlir::Value expr) {
  auto *outPtr = output.columnsForWrite()[col].getForWrite<type>();
  for (uint32_t row = 0; row < input.rows(); row++) {
    auto value = evaluate(expr, input, row);
    *outPtr++ = std::get<typename StoredType<type>::type>(value);
  }
}

std::optional<Batch> ProjectOperator::poll() {
  auto input = child()->poll();
  if (!input) {
    return std::nullopt;
  }

  Batch output(_outputColumnTypes, input->rows());
  for (size_t col = 0; col < _outputColumnTypes.size(); col++) {
    auto val = _expr->getOperand(col);

    switch (_outputColumnTypes[col]) {
#define CASE(t)                                                                \
  case PhysicalColumnType::t:                                                  \
    projectColumn<PhysicalColumnType::t>(*input, output, col, val);            \
    break;

      CASE(INT32)
      CASE(INT64)
      CASE(DOUBLE)
      CASE(STRING)
#undef CASE
    }
  }

  return output;
}

} // namespace execution