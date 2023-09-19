#include "execution/expression/IR/ExpressionDialect.h"
#include "execution/expression/IR/ExpressionOps.h"

using namespace mlir;
using namespace execution::expression;

#include "execution/expression/IR/ExpressionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Expression dialect.
//===----------------------------------------------------------------------===//

void ExpressionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "execution/expression/IR/ExpressionOps.cpp.inc"
      >();
}
