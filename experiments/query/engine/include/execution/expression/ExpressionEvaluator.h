#pragma once

#include "mlir/IR/Operation.h"

#include "execution/expression/ConstantExpression.h"

namespace execution {

ConstantValue evaluate(mlir::Operation *expr, const Batch &batch, uint32_t row);
} // namespace execution
