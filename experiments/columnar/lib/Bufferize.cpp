#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "columnar/Columnar.h"

namespace columnar {

bool TableColumnReadOp::bufferizesToAllocation(mlir::Value v) { return true; }

static std::optional<llvm::Twine> columnReadFuncForType(mlir::TensorType type) {
  auto elemType = type.getElementType();
  if (elemType.isInteger(32)) {
    return "col_table_column_read_int32";
  }

  return std::nullopt;
}

mlir::LogicalResult TableColumnReadOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  auto func = columnReadFuncForType(getType());
  if (!func) {
    return emitOpError("unsupported type") << getType();
  }

  // Allocate an output buffer.
  auto tensorType = getResult().getType();
  auto memrefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  auto buffer =
      rewriter.create<mlir::memref::AllocOp>(getLoc(), memrefType, getSize());

  // Call the runtime function.
  rewriter.create<RuntimeCallOp>(
      getLoc(), mlir::TypeRange{}, rewriter.getStringAttr(*func),
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

static std::optional<llvm::Twine>
chunkAppendFuncForType(mlir::TensorType type) {
  auto elemType = type.getElementType();
  if (elemType.isInteger(32)) {
    return "col_print_chunk_append_int32";
  }

  return std::nullopt;
}

mlir::LogicalResult PrintChunkAppendOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  auto func = chunkAppendFuncForType(getCol().getType());
  if (!func) {
    return emitOpError("unsupported type") << getCol().getType();
  }

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
  mlir::bufferization::replaceOpWithNewBufferizedOp<RuntimeCallOp>(
      rewriter, *this, mlir::TypeRange{}, rewriter.getStringAttr(*func),
      mlir::ValueRange{getChunk(), *col, *sel});
  return mlir::success();
}

} // namespace columnar
