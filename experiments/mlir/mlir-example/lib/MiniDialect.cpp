#include "MiniDialect.h"
#include "MiniOps.h"

using namespace mlir;
using namespace experiments_mlir::mini;

#include "MiniOpsDialect.cpp.inc"

void MiniDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MiniOps.cpp.inc"
      >();
}
