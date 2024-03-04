#include "PhysicalPlanOps.h"
#include "PhysicalPlanDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

#define GET_OP_CLASSES
#include "PhysicalPlanOps.cpp.inc"

namespace physicalplan {

void PackVectorsOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::ValueRange inputs) {
  build(builder, state,
        mlir::TupleType::get(builder.getContext(), inputs.getTypes()), inputs);
}

} // namespace physicalplan