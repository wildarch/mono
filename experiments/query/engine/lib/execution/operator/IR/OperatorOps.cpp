#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/IR/OperatorDialect.h"
#include "execution/operator/IR/OperatorTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/FormatVariadic.h"

#define GET_OP_CLASSES
#include "execution/operator/IR/OperatorOps.cpp.inc"

namespace execution {
namespace qoperator {

mlir::LogicalResult ScanParquetOp::verify() {
  // TODO: duplicate check?
  auto namedColumns = getColumns().size();
  auto typedColumns = getType().getColumns().size();
  if (namedColumns != typedColumns) {
    emitOpError(llvm::formatv(
        "named {} columns to read, but output table has {} columns",
        namedColumns, typedColumns));
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult FilterOp::verify() {
  // Test that the predicate block arguments match the child column types
  auto columnTypes = getChild().getType().getColumns();
  auto predicateBlockArgTypes = getPredicate().getArgumentTypes();
  if (columnTypes.size() != predicateBlockArgTypes.size()) {
    emitOpError("predicate does not have an arg for every column");
    return mlir::failure();
  }

  for (auto [colType, argType] :
       llvm::zip_equal(columnTypes, predicateBlockArgTypes)) {
    if (colType != argType) {
      emitOpError("column types do not match predicate argument types");
      return mlir::failure();
    }
  }

  return mlir::success();
}

} // namespace qoperator
} // namespace execution