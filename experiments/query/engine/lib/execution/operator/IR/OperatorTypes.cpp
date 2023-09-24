#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#include "execution/operator/IR/OperatorDialect.h"
#include "execution/operator/IR/OperatorTypes.h"

#define GET_TYPEDEF_CLASSES
#include "execution/operator/IR/OperatorOpsTypes.cpp.inc"

namespace execution {
namespace qoperator {

void OperatorDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "execution/operator/IR/OperatorOpsTypes.cpp.inc"
      >();
}

} // namespace qoperator
} // namespace execution