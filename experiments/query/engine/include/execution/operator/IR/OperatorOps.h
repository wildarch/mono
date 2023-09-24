#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "execution/operator/IR/OperatorTypes.h"

#define GET_OP_CLASSES
#include "execution/operator/IR/OperatorOps.h.inc"