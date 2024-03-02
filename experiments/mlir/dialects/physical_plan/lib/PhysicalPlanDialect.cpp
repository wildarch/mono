#include "mlir/IR/DialectImplementation.h"

#include "PhysicalPlanOps.h"
#include "PhysicalPlanDialect.h"

using namespace mlir;
using namespace physicalplan;

#include "PhysicalPlanOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PhysicalPlan dialect.
//===----------------------------------------------------------------------===//

void PhysicalPlanDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PhysicalPlanOps.cpp.inc"
      >();
}

Type PhysicalPlanDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(), "unknown type");
  return nullptr;
}

void PhysicalPlanDialect::printType(Type ty, DialectAsmPrinter &p) const {
  p.printKeywordOrString("UNKNOWN_TYPE");
}