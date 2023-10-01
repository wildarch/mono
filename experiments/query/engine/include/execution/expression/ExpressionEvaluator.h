#pragma once

#include "execution/Batch.h"
#include "execution/Common.h"
#include "mlir/IR/Operation.h"
#include <variant>

namespace execution {

using AnyValue = std::variant<bool, int32_t, int64_t, double, SmallString>;

// For use with std::visit
template <class... Ts> struct AnyValueVisitor : Ts... {
  using Ts::operator()...;
};
template <class... Ts> AnyValueVisitor(Ts...) -> AnyValueVisitor<Ts...>;

AnyValue evaluate(mlir::Value val, const Batch &batch, uint32_t row);
AnyValue evaluate(mlir::Operation *expr, const Batch &batch, uint32_t row);
} // namespace execution
