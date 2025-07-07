#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ValueRange.h>

#include "columnar/Columnar.h"

namespace columnar {

bool TableColumnReadOp::bufferizesToAllocation(mlir::Value v) { return true; }

static std::optional<llvm::Twine> columnReadFuncForType(mlir::TensorType type) {
  auto elemType = type.getElementType();
  if (elemType.isInteger(32)) {
    return "col_table_column_read_int32";
  } else if (llvm::isa<ByteArrayType>(elemType)) {
    return "col_table_column_read_byte_array";
  }

  return std::nullopt;
}

static bool isVariableLength(mlir::Type t) {
  return llvm::isa<ByteArrayType>(t);
}

mlir::LogicalResult TableColumnReadOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  auto func = columnReadFuncForType(getType());
  if (!func) {
    return emitOpError("unsupported type ") << getType();
  }

  // Allocate an output buffer.
  auto tensorType = getResult().getType();
  auto memrefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  auto buffer =
      rewriter.create<mlir::memref::AllocOp>(getLoc(), memrefType, getSize());

  llvm::SmallVector<mlir::Value> callArgs{getHandle(), getRowGroup(), getSkip(),
                                          getSize(), buffer};
  if (isVariableLength(getType().getElementType())) {
    // Reading values of variable size, requiring heap allocation.
    // We add the context for access to the allocator.
    callArgs.push_back(getCtx());
  }

  // Call the runtime function.
  rewriter.create<RuntimeCallOp>(getLoc(), mlir::TypeRange{},
                                 rewriter.getStringAttr(*func), callArgs);
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
  } else if (llvm::isa<ByteArrayType>(elemType)) {
    return "col_print_chunk_append_string";
  }

  return std::nullopt;
}

mlir::LogicalResult PrintChunkAppendOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  auto func = chunkAppendFuncForType(getCol().getType());
  if (!func) {
    return emitOpError("unsupported type ") << getCol().getType();
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

// HashOp
bool HashOp::bufferizesToAllocation(mlir::Value v) { return true; }

bool HashOp::bufferizesToMemoryRead(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return true;
}

bool HashOp::bufferizesToMemoryWrite(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return false;
}

mlir::bufferization::AliasingValueList
HashOp::getAliasingValues(mlir::OpOperand &oper,
                          const mlir::bufferization::AnalysisState &state) {
  return {};
}

static std::optional<llvm::Twine> hashFuncForType(mlir::TensorType type) {
  auto elemType = type.getElementType();
  if (elemType.isInteger(32)) {
    return "col_hash_int32";
  } else if (elemType.isInteger(64)) {
    return "col_hash_int64";
  }

  return std::nullopt;
}

mlir::LogicalResult
HashOp::bufferize(mlir::RewriterBase &rewriter,
                  const mlir::bufferization::BufferizationOptions &opts) {
  auto func = hashFuncForType(getType());
  if (!func) {
    return emitOpError("unsupported type ") << getType();
  }

  // Get buffers for base selection vector and value.
  auto base = mlir::bufferization::getBuffer(rewriter, getBase(), opts);
  if (mlir::failed(base)) {
    return mlir::failure();
  }

  auto sel = mlir::bufferization::getBuffer(rewriter, getSel(), opts);
  if (mlir::failed(sel)) {
    return mlir::failure();
  }

  auto value = mlir::bufferization::getBuffer(rewriter, getValue(), opts);
  if (mlir::failed(value)) {
    return mlir::failure();
  }

  // Get size
  auto dimOp = rewriter.create<mlir::memref::DimOp>(getLoc(), *sel, 0);

  // Allocate an output buffer.
  auto tensorType = getResult().getType();
  auto memrefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  // TODO: use createAlloc.
  auto buffer = rewriter.create<mlir::memref::AllocOp>(getLoc(), memrefType,
                                                       mlir::ValueRange{dimOp});

  llvm::SmallVector<mlir::Value> callArgs{*base, *sel, *value, buffer};

  // Call the runtime function.
  rewriter.create<RuntimeCallOp>(getLoc(), mlir::TypeRange{},
                                 rewriter.getStringAttr(*func), callArgs);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     mlir::ValueRange{buffer});
  return mlir::success();
}

// TupleBufferInsertOp
bool TupleBufferInsertOp::bufferizesToAllocation(mlir::Value v) { return true; }

bool TupleBufferInsertOp::bufferizesToMemoryRead(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return true;
}

bool TupleBufferInsertOp::bufferizesToMemoryWrite(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return false;
}

mlir::bufferization::AliasingValueList TupleBufferInsertOp::getAliasingValues(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return {};
}

mlir::LogicalResult TupleBufferInsertOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &opts) {
  auto hashes = mlir::bufferization::getBuffer(rewriter, getHashes(), opts);
  if (mlir::failed(hashes)) {
    return mlir::failure();
  }

  // Get size
  auto dimOp = rewriter.create<mlir::memref::DimOp>(getLoc(), *hashes, 0);

  // Allocate an output buffer.
  auto tensorType = getResult().getType();
  auto memrefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  // TODO: use createAlloc.
  auto buffer = rewriter.create<mlir::memref::AllocOp>(getLoc(), memrefType,
                                                       mlir::ValueRange{dimOp});

  llvm::SmallVector<mlir::Value> callArgs{getBuffer(), *hashes, buffer};

  // Call the runtime function.
  rewriter.create<RuntimeCallOp>(getLoc(), mlir::TypeRange{},
                                 "col_tuple_buffer_insert", callArgs);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     mlir::ValueRange{buffer});
  return mlir::success();
}

static std::optional<llvm::Twine> scatterFuncForType(mlir::TensorType type) {
  auto elemType = type.getElementType();
  if (elemType.isInteger(32)) {
    return "col_scatter_int32";
  } else if (elemType.isInteger(64)) {
    return "col_scatter_int64";
  } else if (auto byteArrayType =
                 mlir::dyn_cast<columnar::ByteArrayType>(elemType)) {
    return "col_scatter_byte_array";
  }

  return std::nullopt;
}

// ScatterOp
bool ScatterOp::bufferizesToMemoryRead(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return true;
}

bool ScatterOp::bufferizesToMemoryWrite(
    mlir::OpOperand &oper, const mlir::bufferization::AnalysisState &state) {
  return false;
}

mlir::bufferization::AliasingValueList
ScatterOp::getAliasingValues(mlir::OpOperand &oper,
                             const mlir::bufferization::AnalysisState &state) {
  return {};
}

mlir::LogicalResult
ScatterOp::bufferize(mlir::RewriterBase &rewriter,
                     const mlir::bufferization::BufferizationOptions &opts) {
  auto valueType = mlir::cast<mlir::TensorType>(getValue().getType());
  auto func = scatterFuncForType(valueType);
  if (!func) {
    return emitOpError("unsupported type ") << valueType;
  }

  // Get buffers for all tensor inputs
  auto sel = mlir::bufferization::getBuffer(rewriter, getSel(), opts);
  if (mlir::failed(sel)) {
    return mlir::failure();
  }

  auto value = mlir::bufferization::getBuffer(rewriter, getValue(), opts);
  if (mlir::failed(value)) {
    return mlir::failure();
  }

  auto dest = mlir::bufferization::getBuffer(rewriter, getDest(), opts);
  if (mlir::failed(dest)) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> callArgs{*sel, *value, *dest};

  // Call the runtime function
  auto nextOffset = rewriter.create<RuntimeCallOp>(
      getLoc(), mlir::TypeRange{}, rewriter.getStringAttr(*func), callArgs);

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     nextOffset.getResults());
  return mlir::success();
}

} // namespace columnar
