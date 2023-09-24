#include "execution/operator/IR/OperatorDialect.h"
#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/IR/OperatorTypes.h"

using namespace mlir;
using namespace execution::qoperator;

#include "execution/operator/IR/OperatorOpsDialect.cpp.inc"

void OperatorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "execution/operator/IR/OperatorOps.cpp.inc"
      >();

  registerTypes();
}