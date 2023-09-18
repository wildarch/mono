#pragma once

#include "execution/Batch.h"
#include "execution/Common.h"
#include "execution/expression/Expression.h"

namespace execution {

class ColumnExpression : public Expression {
private:
  ColumnIdx _idx;
  PhysicalColumnType _type;

public:
  inline ColumnExpression(ColumnIdx idx, PhysicalColumnType type)
      : Expression(ExpressionKind::COLUMN), _idx(idx), _type(type) {}

  inline auto idx() const { return _idx; }
  inline auto type() const { return _type; }
};

} // namespace execution