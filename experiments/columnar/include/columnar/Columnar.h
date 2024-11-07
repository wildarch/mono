#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

#include "columnar/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "columnar/Types.h.inc"

#define GET_OP_CLASSES
#include "columnar/Ops.h.inc"

namespace columnar {} // namespace columnar