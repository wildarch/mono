#pragma once

#include "execution/expression/ConstantExpression.h"

namespace execution {

ConstantValue evaluate(const Expression &expr, const Batch &batch,
                       uint32_t row);

}
