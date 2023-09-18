#pragma once

#include <memory>

namespace execution {

enum class ExpressionKind {
  BINARY_OPERATOR,
  COLUMN,
  CONSTANT,
};

class Expression {
private:
  ExpressionKind _kind;

protected:
  inline Expression(ExpressionKind kind) : _kind(kind) {}

public:
  inline ExpressionKind kind() const { return _kind; }
};

using ExpressionPtr = std::shared_ptr<Expression>;

} // namespace execution