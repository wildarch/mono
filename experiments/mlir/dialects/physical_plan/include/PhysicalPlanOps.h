#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "PhysicalPlanTypes.h"

#define GET_OP_CLASSES
#include "PhysicalPlanOps.h.inc"