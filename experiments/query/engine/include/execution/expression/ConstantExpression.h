#pragma once

#include <cstdint>
#include <variant>

#include "execution/Batch.h"
#include "execution/Common.h"
#include "execution/expression/Expression.h"

namespace execution {

using ConstantValue = std::variant<bool, int32_t, double, SmallString>;

template <class... Ts> struct ConstantVisitor : Ts... {
  using Ts::operator()...;
};

class ConstantExpression : public Expression {
private:
  ConstantValue _value;

public:
  inline ConstantExpression(ConstantValue value)
      : Expression(ExpressionKind::CONSTANT), _value(value) {}

  inline ConstantValue value() const { return _value; }
};

} // namespace execution