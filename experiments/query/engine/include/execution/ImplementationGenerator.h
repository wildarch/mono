#pragma once

#include "mlir/IR/Operation.h"

#include "execution/operator/impl/Operator.h"

namespace execution {

OperatorPtr generateImplementation(mlir::Operation *root);

}