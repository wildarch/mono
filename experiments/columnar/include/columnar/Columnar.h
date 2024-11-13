#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "columnar/Dialect.h.inc"

#include "columnar/Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "columnar/Attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "columnar/Types.h.inc"

#define GET_OP_CLASSES
#include "columnar/Ops.h.inc"

namespace columnar {

#define GEN_PASS_DECL
#include "columnar/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "columnar/Passes.h.inc"

/** Element type for columns produced by a COUNT aggregator. */
mlir::Type getCountElementType(mlir::MLIRContext *ctx);

} // namespace columnar