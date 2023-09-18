#pragma once

#include "execution/expression/BinaryOperatorExpression.h"
#include "execution/expression/ColumnExpression.h"
#include "execution/expression/ConstantExpression.h"
#include "execution/expression/Expression.h"

namespace execution {

template <class... Ts> struct ExpressionVisitor : Ts... {
  using Ts::operator()...;

  auto visit(const Expression &expr) {
    switch (expr.kind()) {
    case ExpressionKind::BINARY_OPERATOR:
      return (*this)(static_cast<const BinaryOperatorExpression &>(expr));
    case ExpressionKind::COLUMN:
      return (*this)(static_cast<const ColumnExpression &>(expr));
    case ExpressionKind::CONSTANT:
      return (*this)(static_cast<const ConstantExpression &>(expr));
    }

    throw std::logic_error("unvalid enum");
  }
};
template <class... Ts> ExpressionVisitor(Ts...) -> ExpressionVisitor<Ts...>;

} // namespace execution