#include "columnar/Columnar.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "columnar/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "columnar/Types.cpp.inc"

#define GET_OP_CLASSES
#include "columnar/Ops.cpp.inc"

namespace columnar {

void ColumnarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "columnar/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "columnar/Types.cpp.inc"
      >();
}

} // namespace columnar