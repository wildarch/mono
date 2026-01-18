#include <cassert>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

#include "columnar/Columnar.h"

namespace columnar {

#define GEN_PASS_DEF_LOWERPIPELINES
#include "columnar/Passes.h.inc"

namespace {

class LowerPipelines : public impl::LowerPipelinesBase<LowerPipelines> {
public:
  using impl::LowerPipelinesBase<LowerPipelines>::LowerPipelinesBase;

  void runOnOperation() final;
};

class ColumnTypeConverter : public mlir::TypeConverter {
public:
  ColumnTypeConverter();
};

} // namespace

// Marks all instances of this type as valid without any conversion.
template <typename T> static mlir::Type markTypeAllowed(T t) { return t; }

ColumnTypeConverter::ColumnTypeConverter() {
  addConversion(markTypeAllowed<mlir::FloatType>);
  addConversion(markTypeAllowed<mlir::IntegerType>);
  addConversion(
      [](SelectType t) { return mlir::IndexType::get(t.getContext()); });
  addConversion([this](ColumnType t) {
    mlir::Type elementType = convertType(t.getElementType());
    if (!elementType) {
      return mlir::RankedTensorType();
    }

    return mlir::RankedTensorType::get(
        llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic}, elementType);
  });
  addConversion(
      [](StringType t) { return ByteArrayType::get(t.getContext()); });
}

static mlir::RankedTensorType getSelectionVectorType(mlir::OpBuilder &builder) {
  return mlir::RankedTensorType::get(
      llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic},
      builder.getIndexType());
}

// Generate the iota selection vector (0..size)
static mlir::tensor::GenerateOp
buildIotaSelectionVector(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value size) {
  return builder.create<mlir::tensor::GenerateOp>(
      loc, getSelectionVectorType(builder), mlir::ValueRange{size},
      [](mlir::OpBuilder &builder, mlir::Location loc,
         mlir::ValueRange indices) {
        builder.create<mlir::tensor::YieldOp>(loc, indices[0]);
      });
}

// ReadTableOp
mlir::LogicalResult
ReadTableOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                             llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  // Open scanner
  auto tablePath =
      builder.create<ConstantStringOp>(getLoc(), getTable().getPath());
  auto scannerOp = builder.create<RuntimeCallOp>(
      getLoc(), builder.getType<ScannerHandleType>(),
      builder.getStringAttr("col_table_scanner_open"),
      mlir::ValueRange{tablePath});
  newGlobals.push_back(scannerOp.getResult(0));

  // Open columns
  for (auto col : getColumnsToRead()) {
    auto colIdx = builder.create<mlir::arith::ConstantOp>(
        getLoc(), builder.getI32Type(),
        builder.getI32IntegerAttr(col.getIdx()));
    auto columnOp = builder.create<RuntimeCallOp>(
        getLoc(), builder.getType<ColumnHandleType>(),
        builder.getStringAttr("col_table_column_open"),
        mlir::ValueRange{tablePath, colIdx});
    newGlobals.push_back(columnOp->getResult(0));
  }

  return mlir::success();
}

mlir::LogicalResult ReadTableOp::lowerGlobalClose(mlir::OpBuilder &builder,
                                                  mlir::ValueRange globals) {
  // TODO: close the scanner and columns
  return mlir::success();
}

mlir::LogicalResult ReadTableOp::lowerBody(LowerBodyCtx &ctx,
                                           mlir::OpBuilder &builder) {
  auto scanner = ctx.globals[0];
  auto columns = ctx.globals.drop_front();

  // Claim a chunk of rows to read
  auto claimOp = builder.create<RuntimeCallOp>(
      getLoc(),
      mlir::TypeRange{builder.getI32Type(), builder.getI32Type(),
                      builder.getIndexType()},
      builder.getStringAttr("col_table_scanner_claim_chunk"), scanner);
  auto rowGroup = claimOp->getResult(0);
  auto skip = claimOp->getResult(1);
  auto size = claimOp->getResult(2);

  // If the chunk has size > 0, there may be more to read.
  auto zeroOp = builder.create<mlir::arith::ConstantOp>(
      getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  auto haveRowsOp = builder.create<mlir::arith::CmpIOp>(
      getLoc(), mlir::arith::CmpIPredicate::ugt, size, zeroOp);
  ctx.haveMore.push_back(haveRowsOp);

  auto selOp = buildIotaSelectionVector(builder, getLoc(), size);
  ctx.results.push_back(selOp);

  // Read the columns
  for (auto [col, type] : llvm::zip_equal(columns, getCol().getTypes())) {
    mlir::Type tensorType = ctx.typeConverter.convertType(type);
    if (!tensorType) {
      return emitError("cannot convert column type: ") << type;
    }

    auto readOp = builder.create<TableColumnReadOp>(
        getLoc(), tensorType, ctx.pipelineCtx, col, rowGroup, skip, size);
    ctx.results.push_back(readOp);
  }

  return mlir::success();
}

// QueryOutputOp
mlir::LogicalResult
QueryOutputOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                               llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  auto printOp = builder.create<RuntimeCallOp>(
      getLoc(), builder.getType<PrintHandleType>(),
      builder.getStringAttr("col_print_open"), mlir::ValueRange{});
  newGlobals.push_back(printOp.getResult(0));
  return mlir::success();
}

mlir::LogicalResult QueryOutputOp::lowerGlobalClose(mlir::OpBuilder &builder,
                                                    mlir::ValueRange globals) {
  // TODO: Close result printer
  return mlir::success();
}

mlir::LogicalResult QueryOutputOp::lowerBody(LowerBodyCtx &ctx,
                                             mlir::OpBuilder &builder) {
  Adaptor adaptor(ctx.operands, *this);
  auto sel = adaptor.getSel();
  if (!sel) {
    return mlir::failure();
  }

  auto handle = ctx.globals[0];

  // New chunk
  auto nrows = builder.create<mlir::tensor::DimOp>(getLoc(), sel, 0);
  auto allocOp = builder.create<RuntimeCallOp>(
      getLoc(), builder.getType<PrintChunkType>(),
      builder.getStringAttr("col_print_chunk_alloc"), mlir::ValueRange{nrows});
  auto chunk = allocOp.getResult(0);

  // Append columns
  for (auto col : adaptor.getColumns()) {
    builder.create<PrintChunkAppendOp>(getLoc(), chunk, col, sel);
  }

  // Write chunk
  builder.create<RuntimeCallOp>(getLoc(), mlir::TypeRange{},
                                builder.getStringAttr("col_print_write"),
                                mlir::ValueRange{handle, chunk});
  return mlir::success();
}

// HashJoinCollectOp
mlir::LogicalResult HashJoinCollectOp::lowerLocalOpen(
    mlir::OpBuilder &builder, mlir::ValueRange globals,
    llvm::SmallVectorImpl<mlir::Value> &newLocals) {
  auto tupleType = getBufferType().getTupleType();
  auto sizeOp = builder.create<TypeSizeOp>(getLoc(), tupleType);
  auto alignOp = builder.create<TypeAlignOp>(getLoc(), tupleType);

  // The state type
  auto localType = builder.getType<TupleBufferLocalType>(tupleType);
  auto allocOp = builder.create<RuntimeCallOp>(
      getLoc(), mlir::TypeRange{localType},
      builder.getStringAttr("col_tuple_buffer_local_alloc"),
      mlir::ValueRange{sizeOp, alignOp});
  newLocals.push_back(allocOp->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
HashJoinCollectOp::lowerLocalClose(mlir::OpBuilder &builder,
                                   mlir::ValueRange globals,
                                   mlir::ValueRange locals) {
  // Merge local buffer to global one.
  auto localBuffer = locals[0];
  auto globalBuffer =
      builder.create<GlobalReadOp>(getLoc(), getBufferType(), getBuffer());
  builder.create<RuntimeCallOp>(getLoc(), mlir::TypeRange{},
                                builder.getStringAttr("col_tuple_buffer_merge"),
                                mlir::ValueRange{globalBuffer, localBuffer});
  return mlir::success();
}

mlir::LogicalResult HashJoinCollectOp::lowerBody(LowerBodyCtx &ctx,
                                                 mlir::OpBuilder &builder) {
  Adaptor adaptor(ctx.operands, *this);

  // Hash the columns.
  auto nrows = builder.create<mlir::tensor::DimOp>(
      getLoc(), adaptor.getKeySel().front(), 0);
  auto hashType = mlir::RankedTensorType::get(
      llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic},
      builder.getI64Type());

  if (adaptor.getKeys().empty()) {
    return emitOpError("Need at least one 1 key");
  }

  auto hashOp =
      builder.create<HashOp>(getLoc(), hashType, adaptor.getKeySel().front(),
                             adaptor.getKeys().front());
  if (adaptor.getKeys().size() > 1) {
    return emitOpError("TODO: hash combine");
  }

  // Allocate space for the entries (picking partitions based on the hash).
  auto localBuffer = ctx.locals[0];
  auto allocOp =
      builder.create<TupleBufferInsertOp>(getLoc(), localBuffer, hashOp);

  auto allocator =
      builder
          .create<RuntimeCallOp>(
              getLoc(), mlir::TypeRange{builder.getType<AllocatorType>()},
              builder.getStringAttr("col_tuple_buffer_local_get_allocator"),
              mlir::ValueRange{localBuffer})
          ->getResult(0);

  // Scatter the columns.
  llvm::SmallVector<mlir::Value> columnSel;
  llvm::append_range(columnSel, adaptor.getKeySel());
  llvm::append_range(columnSel, adaptor.getValueSel());
  llvm::SmallVector<mlir::Value> columnValues;
  llvm::append_range(columnValues, adaptor.getKeys());
  llvm::append_range(columnValues, adaptor.getValues());
  assert(columnSel.size() == columnValues.size());
  auto structType = getBufferType().getTupleType();
  for (auto [f, sel, val] : llvm::enumerate(columnSel, columnValues)) {
    // First field contains the hash.
    auto field = f + 1;
    // Offset the base pointers to get pointers to the field we want to write.
    auto fieldType = structType.getFieldTypes()[field];
    auto resultType = tensorColOf(PointerType::get(fieldType));
    auto genOp = builder.create<mlir::tensor::GenerateOp>(
        getLoc(), resultType, mlir::ValueRange{nrows},
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange indices) {
          auto ptr =
              builder.create<mlir::tensor::ExtractOp>(loc, allocOp, indices);
          auto offset = builder.create<GetFieldPtrOp>(loc, ptr, field);
          builder.create<mlir::tensor::YieldOp>(loc, offset);
        });
    builder.create<ScatterOp>(getLoc(), sel, val, genOp, allocator);
  }

  return mlir::success();
}

// TODO: HashJoinBuildOp
// 1. GLOBAL: Allocate directory, size based on the total nr. of tuples
// 2. Count number of tuples per hash (write to directory)
// 3. Apply exclusive prefix sum over directory, to find where each directory
// starts.
// 4. Copy tuples to final destination (and update corresponding directory slot)

static void unpackStructPointer(mlir::Value v, mlir::OpBuilder &builder,
                                llvm::SmallVectorImpl<mlir::Value> &out) {
  auto ptrType = llvm::cast<PointerType>(v.getType());
  auto structType = llvm::cast<StructType>(ptrType.getPointee());
  for (auto i : llvm::seq(structType.getFieldTypes().size())) {
    out.emplace_back(builder.create<GetStructElementOp>(v.getLoc(), v, i));
  }
}

static mlir::LogicalResult lowerPipeline(mlir::TypeConverter &typeConverter,
                                         mlir::IRRewriter &rewriter,
                                         PipelineOp pipelineOp) {
  auto lowerOp = rewriter.create<PipelineLowOp>(pipelineOp->getLoc());

  // Blocks
  auto &globalOpenBlock = lowerOp.getGlobalOpen().emplaceBlock();
  auto &localOpenBlock = lowerOp.getLocalOpen().emplaceBlock();
  auto &bodyBlock = lowerOp.getBody().emplaceBlock();
  auto &localCloseBlock = lowerOp.getLocalClose().emplaceBlock();
  auto &globalCloseBlock = lowerOp.getGlobalClose().emplaceBlock();

  // All ops must implement the interface
  llvm::SmallVector<LowerPipelineOpInterface> toLower;
  for (auto &op : pipelineOp.getBody().front()) {
    auto iface = llvm::dyn_cast<LowerPipelineOpInterface>(op);
    if (!iface) {
      return op.emitOpError("does not implement LowerPipelineOpInterface");
    }

    toLower.push_back(iface);
  }

  // Pipeline context available in the body.
  bodyBlock.addArgument(rewriter.getType<PipelineContextType>(),
                        pipelineOp.getLoc());

  // Number of globals opened per op
  llvm::SmallVector<unsigned int> globalsPerOp;
  // Number of locals opened per op
  llvm::SmallVector<unsigned int> localsPerOp;

  // Global open
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&globalOpenBlock);
    llvm::SmallVector<mlir::Value> globals;
    for (auto op : toLower) {
      llvm::SmallVector<mlir::Value> newGlobals;
      if (mlir::failed(op.lowerGlobalOpen(rewriter, newGlobals))) {
        return mlir::failure();
      }

      globals.append(newGlobals);
      globalsPerOp.push_back(newGlobals.size());
    }

    auto globalStructOp =
        rewriter.create<AllocStructOp>(pipelineOp.getLoc(), globals);
    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(),
                                        mlir::ValueRange{globalStructOp});

    // Globals are available in all blocks
    localOpenBlock.addArgument(globalStructOp.getType(),
                               globalStructOp.getLoc());
    bodyBlock.addArgument(globalStructOp.getType(), globalStructOp.getLoc());
    localCloseBlock.addArgument(globalStructOp.getType(),
                                globalStructOp.getLoc());
    globalCloseBlock.addArgument(globalStructOp.getType(),
                                 globalStructOp.getLoc());
  }

  // Global free
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&globalCloseBlock);

    llvm::SmallVector<mlir::Value> globalArgs;
    unpackStructPointer(globalCloseBlock.getArgument(0), rewriter, globalArgs);
    auto args = llvm::ArrayRef<mlir::Value>(globalArgs);

    for (auto [op, numGlobals] : llvm::zip_equal(toLower, globalsPerOp)) {
      auto opArgs = args.take_front(numGlobals);
      args = args.drop_front(numGlobals);
      if (mlir::failed(op.lowerGlobalClose(rewriter, opArgs))) {
        return mlir::failure();
      }
    }

    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(),
                                        mlir::ValueRange{});
  }

  // Local open
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&localOpenBlock);

    llvm::SmallVector<mlir::Value> globalArgs;
    unpackStructPointer(localOpenBlock.getArgument(0), rewriter, globalArgs);
    auto globalArgsLeft = llvm::ArrayRef<mlir::Value>(globalArgs);

    llvm::SmallVector<mlir::Value> locals;
    for (auto [op, numGlobals] : llvm::zip_equal(toLower, globalsPerOp)) {
      auto globals = globalArgsLeft.take_front(numGlobals);
      globalArgsLeft = globalArgsLeft.drop_front(numGlobals);

      llvm::SmallVector<mlir::Value> newLocals;
      if (mlir::failed(op.lowerLocalOpen(rewriter, globals, newLocals))) {
        return mlir::failure();
      }

      locals.append(newLocals);
      localsPerOp.push_back(newLocals.size());
    }

    auto localStructOp =
        rewriter.create<AllocStructOp>(pipelineOp.getLoc(), locals);
    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(),
                                        mlir::ValueRange{localStructOp});

    // Locals are available in body and local close
    bodyBlock.addArgument(localStructOp.getType(), localStructOp.getLoc());
    localCloseBlock.addArgument(localStructOp.getType(),
                                localStructOp.getLoc());
  }

  // Local close
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&localCloseBlock);

    llvm::SmallVector<mlir::Value> globalArgs;
    unpackStructPointer(localCloseBlock.getArgument(0), rewriter, globalArgs);
    auto globalArgsLeft = llvm::ArrayRef<mlir::Value>(globalArgs);

    llvm::SmallVector<mlir::Value> localArgs;
    unpackStructPointer(localCloseBlock.getArgument(1), rewriter, localArgs);
    auto localArgsLeft = llvm::ArrayRef<mlir::Value>(localArgs);

    for (auto [op, numGlobals, numLocals] :
         llvm::zip_equal(toLower, globalsPerOp, localsPerOp)) {
      auto globals = globalArgsLeft.take_front(numGlobals);
      globalArgsLeft = globalArgsLeft.drop_front(numGlobals);
      auto locals = localArgsLeft.take_front(numLocals);
      localArgsLeft = localArgsLeft.drop_front(numLocals);

      if (mlir::failed(op.lowerLocalClose(rewriter, globals, locals))) {
        return mlir::failure();
      }
    }

    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(),
                                        mlir::ValueRange{});
  }

  // Body
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&bodyBlock);

    // Maps op results to the new results in the lowered body.
    mlir::IRMapping mapping;

    // Tracks whether all ops in the body want to be called again.
    llvm::SmallVector<mlir::Value> haveMore;

    llvm::SmallVector<mlir::Value> globalArgs;
    unpackStructPointer(bodyBlock.getArgument(1), rewriter, globalArgs);
    auto globalArgsLeft = llvm::ArrayRef<mlir::Value>(globalArgs);

    llvm::SmallVector<mlir::Value> localArgs;
    unpackStructPointer(bodyBlock.getArgument(2), rewriter, localArgs);
    auto localArgsLeft = llvm::ArrayRef<mlir::Value>(localArgs);

    for (auto [op, numGlobals, numLocals] :
         llvm::zip_equal(toLower, globalsPerOp, localsPerOp)) {
      auto globals = globalArgsLeft.take_front(numGlobals);
      globalArgsLeft = globalArgsLeft.drop_front(numGlobals);
      auto locals = localArgsLeft.take_front(numLocals);
      localArgsLeft = localArgsLeft.drop_front(numLocals);

      llvm::SmallVector<mlir::Value> operands;
      for (auto oper : op->getOperands()) {
        // TODO: catch failures here.
        operands.push_back(mapping.lookupOrDefault(oper));
      }

      LowerBodyCtx ctx{typeConverter, bodyBlock.getArgument(0), globals, locals,
                       operands};
      if (mlir::failed(op.lowerBody(ctx, rewriter))) {
        return mlir::failure();
      }

      // Map results
      mapping.map(op->getResults(), ctx.results);

      // Merge haveMore
      haveMore.append(ctx.haveMore);
    }

    if (haveMore.empty()) {
      return pipelineOp->emitOpError("no ops populate 'haveMore'");
    }

    // Merge all 'haveMore' values
    auto allHaveMore = haveMore[0];
    for (auto v : mlir::ValueRange{haveMore}.drop_front()) {
      allHaveMore =
          rewriter.create<mlir::arith::AndIOp>(v.getLoc(), allHaveMore, v);
    }

    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(),
                                        mlir::ValueRange{allHaveMore});
  }

  rewriter.replaceOp(pipelineOp, lowerOp);
  return mlir::success();
}

void LowerPipelines::runOnOperation() {
  ColumnTypeConverter typeConverter;

  llvm::SmallVector<PipelineOp> pipelineOps(
      getOperation().getOps<PipelineOp>());
  mlir::IRRewriter rewriter(getOperation());
  rewriter.setInsertionPointToEnd(getOperation().getBody());

  bool hadFailure = false;
  for (auto op : pipelineOps) {
    if (mlir::failed(lowerPipeline(typeConverter, rewriter, op))) {
      hadFailure = true;
    }
  }

  if (hadFailure) {
    return signalPassFailure();
  }
}

} // namespace columnar
