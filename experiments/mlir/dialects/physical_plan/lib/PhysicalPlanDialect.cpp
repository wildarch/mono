#include "mlir/IR/DialectImplementation.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanOps.h"
#include "PhysicalPlanTypes.h"

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

  registerTypes();
}