#include "mlir/IR/DialectImplementation.h"

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

Type ExpressionDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(), "unknown type");
  return nullptr;
}

void ExpressionDialect::printType(Type ty, DialectAsmPrinter &p) const {
  p.printKeywordOrString("UNKNOWN_TYPE");
}