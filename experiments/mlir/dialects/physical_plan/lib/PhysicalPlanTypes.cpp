#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanTypes.h"

#define GET_TYPEDEF_CLASSES
#include "PhysicalPlanOpsTypes.cpp.inc"

namespace physicalplan {

void PhysicalPlanDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PhysicalPlanOpsTypes.cpp.inc"
      >();
}

} // namespace physicalplan