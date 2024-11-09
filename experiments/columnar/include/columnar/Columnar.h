#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "columnar/Dialect.h.inc"

#include "columnar/Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "columnar/Attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "columnar/Types.h.inc"

#define GET_OP_CLASSES
#include "columnar/Ops.h.inc"

namespace columnar {

/** Element type for columns produced by a COUNT aggregator. */
mlir::Type getCountElementType(mlir::MLIRContext *ctx);

} // namespace columnar