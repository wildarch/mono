#pragma once

#include "execution/expression/Expression.h"

namespace execution {

enum class BinaryOperator {
  LE_INT32,
};

class BinaryOperatorExpression : public Expression {
private:
  ExpressionPtr _lhs;
  BinaryOperator _op;
  ExpressionPtr _rhs;

public:
  inline BinaryOperatorExpression(ExpressionPtr lhs, BinaryOperator op,
                                  ExpressionPtr rhs)
      : Expression(ExpressionKind::BINARY_OPERATOR), _lhs(lhs), _op(op),
        _rhs(rhs) {}

  inline const auto &lhs() const { return _lhs; }
  inline const auto &op() const { return _op; }
  inline const auto &rhs() const { return _rhs; }
};

} // namespace execution