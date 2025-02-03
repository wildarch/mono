#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "columnar/Columnar.h"

namespace columnar {

bool TableColumnReadOp::bufferizesToAllocation(mlir::Value v) { return true; }

mlir::LogicalResult TableColumnReadOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  // Allocate an output buffer.
  auto tensorType = getResult().getType();
  auto memrefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  auto buffer =
      rewriter.create<mlir::memref::AllocOp>(getLoc(), memrefType, getSize());

  // Call the runtime function.
  auto func = rewriter.getStringAttr("col_table_column_read");
  rewriter.create<RuntimeCallOp>(
      getLoc(), mlir::TypeRange{}, func,
      mlir::ValueRange{getHandle(), getStart(), getSize(), buffer});
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     mlir::ValueRange{buffer});
  return mlir::success();
}

bool PrintChunkAppendOp::bufferizesToMemoryRead(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return true;
}

bool PrintChunkAppendOp::bufferizesToMemoryWrite(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return false;
}

mlir::bufferization::AliasingValueList PrintChunkAppendOp::getAliasingValues(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return {};
}

mlir::LogicalResult PrintChunkAppendOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  // Get buffers for column and selection vector.
  auto col = mlir::bufferization::getBuffer(rewriter, getCol(), opts);
  if (mlir::failed(col)) {
    return mlir::failure();
  }

  auto sel = mlir::bufferization::getBuffer(rewriter, getSel(), opts);
  if (mlir::failed(sel)) {
    return mlir::failure();
  }

  // Call the runtime function.
  auto func = rewriter.getStringAttr("col_table_column_read");
  mlir::bufferization::replaceOpWithNewBufferizedOp<RuntimeCallOp>(
      rewriter, *this, mlir::TypeRange{}, func,
      mlir::ValueRange{getChunk(), *col, *sel});
  return mlir::success();
}

} // namespace columnar