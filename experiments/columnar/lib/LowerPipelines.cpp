#include "columnar/Columnar.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace columnar {

#define GEN_PASS_DEF_LOWERPIPELINES
#include "columnar/Passes.h.inc"

namespace {

class LowerPipelines : public impl::LowerPipelinesBase<LowerPipelines> {
public:
  using impl::LowerPipelinesBase<LowerPipelines>::LowerPipelinesBase;

  void runOnOperation() final;
};

} // namespace

// TODO: Use a proper type converter instead.
static mlir::Type convertElementType(mlir::Type t) {
  if (llvm::isa<SelectType>(t)) {
    return mlir::IndexType::get(t.getContext());
  } else if (llvm::isa<mlir::FloatType>(t)) {
    return t;
  } else if (llvm::isa<mlir::IntegerType>(t)) {
    return t;
  }

  mlir::emitError(mlir::UnknownLoc::get(t.getContext()),
                  "cannot convert element type: ")
      << t;
  return nullptr;
}

static mlir::RankedTensorType convertType(ColumnType t) {
  return mlir::RankedTensorType::get(
      llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic},
      convertElementType(t.getElementType()));
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
    auto colPath = builder.create<ConstantStringOp>(getLoc(), col.getPath());
    auto columnOp = builder.create<RuntimeCallOp>(
        getLoc(), builder.getType<ColumnHandleType>(),
        builder.getStringAttr("col_table_column_open"),
        mlir::ValueRange{colPath});
    newGlobals.push_back(columnOp->getResult(0));
  }

  return mlir::success();
}

mlir::LogicalResult ReadTableOp::lowerGlobalClose(mlir::OpBuilder &builder,
                                                  mlir::ValueRange globals) {
  // TODO: close the scanner and columns
  return mlir::success();
}

mlir::LogicalResult
ReadTableOp::lowerBody(mlir::OpBuilder &builder, mlir::ValueRange globals,
                       mlir::ValueRange operands,
                       llvm::SmallVectorImpl<mlir::Value> &results,
                       llvm::SmallVectorImpl<mlir::Value> &haveMore) {
  auto scanner = globals[0];
  auto columns = globals.drop_front();

  // Claim a chunk of rows to read
  auto claimOp = builder.create<RuntimeCallOp>(
      getLoc(), mlir::TypeRange{builder.getIndexType(), builder.getIndexType()},
      builder.getStringAttr("col_table_scanner_claim_chunk"), scanner);
  auto start = claimOp->getResult(0);
  auto size = claimOp->getResult(1);

  // If the chunk has size > 0, there may be more to read.
  auto zeroOp = builder.create<mlir::arith::ConstantOp>(
      getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  auto haveRowsOp = builder.create<mlir::arith::CmpIOp>(
      getLoc(), mlir::arith::CmpIPredicate::ugt, size, zeroOp);
  haveMore.push_back(haveRowsOp);

  auto selOp = buildIotaSelectionVector(builder, getLoc(), size);
  results.push_back(selOp);

  // Read the columns
  for (auto [col, type] : llvm::zip_equal(columns, getCol().getTypes())) {
    auto tensorType = convertType(llvm::cast<ColumnType>(type));
    if (!tensorType) {
      return mlir::failure();
    }

    auto readOp = builder.create<TableColumnReadOp>(getLoc(), tensorType, col,
                                                    start, size);
    results.push_back(readOp);
  }

  return mlir::success();
}

// PrintOp
mlir::LogicalResult
PrintOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                         llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  auto printOp = builder.create<RuntimeCallOp>(
      getLoc(), builder.getType<PrintHandleType>(),
      builder.getStringAttr("col_print_open"), mlir::ValueRange{});
  newGlobals.push_back(printOp.getResult(0));
  return mlir::success();
}

mlir::LogicalResult PrintOp::lowerGlobalClose(mlir::OpBuilder &builder,
                                              mlir::ValueRange globals) {
  // TODO: Close result printer
  return mlir::success();
}

mlir::LogicalResult
PrintOp::lowerBody(mlir::OpBuilder &builder, mlir::ValueRange globals,
                   mlir::ValueRange operands,
                   llvm::SmallVectorImpl<mlir::Value> &results,
                   llvm::SmallVectorImpl<mlir::Value> &haveMore) {
  Adaptor adaptor(operands, *this);
  auto sel = adaptor.getSel();

  auto handle = globals[0];

  // New chunk
  auto nrows = builder.create<mlir::tensor::DimOp>(getLoc(), sel, 0);
  auto allocOp = builder.create<RuntimeCallOp>(
      getLoc(), builder.getType<PrintChunkType>(),
      builder.getStringAttr("col_print_chunk_alloc"), mlir::ValueRange{nrows});
  auto chunk = allocOp.getResult(0);

  // Append columns
  for (auto input : adaptor.getInputs()) {
    builder.create<PrintChunkAppendOp>(getLoc(), chunk, input, sel);
  }

  // Write chunk
  builder.create<RuntimeCallOp>(getLoc(), mlir::TypeRange{},
                                builder.getStringAttr("col_print_write"),
                                mlir::ValueRange{handle, chunk});
  return mlir::success();
}

static void unpackStructPointer(mlir::Value v, mlir::OpBuilder &builder,
                                llvm::SmallVectorImpl<mlir::Value> &out) {
  auto ptrType = llvm::cast<PointerType>(v.getType());
  auto structType = llvm::cast<StructType>(ptrType.getPointee());
  for (auto i : llvm::seq(structType.getFieldTypes().size())) {
    out.emplace_back(builder.create<GetStructElementOp>(v.getLoc(), v, i));
  }
}

static mlir::LogicalResult lowerPipeline(mlir::IRRewriter &rewriter,
                                         PipelineOp pipelineOp) {
  auto lowerOp = rewriter.create<PipelineLowOp>(pipelineOp->getLoc());

  // Blocks
  auto &globalOpenBlock = lowerOp.getGlobalOpen().emplaceBlock();
  auto &bodyBlock = lowerOp.getBody().emplaceBlock();
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

  // Number of globals opened per op
  llvm::SmallVector<unsigned int> globalsPerOp;

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
    bodyBlock.addArgument(globalStructOp.getType(), globalStructOp.getLoc());
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

  // TODO: local open
  // TODO: local close

  // Body
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&bodyBlock);

    // Maps op results to the new results in the lowered body.
    mlir::IRMapping mapping;

    // Tracks whether all ops in the body want to be called again.
    llvm::SmallVector<mlir::Value> haveMore;

    llvm::SmallVector<mlir::Value> globalArgs;
    unpackStructPointer(bodyBlock.getArgument(0), rewriter, globalArgs);
    auto args = llvm::ArrayRef<mlir::Value>(globalArgs);

    for (auto [op, numGlobals] : llvm::zip_equal(toLower, globalsPerOp)) {
      auto globals = args.take_front(numGlobals);
      args = args.drop_front(numGlobals);

      llvm::SmallVector<mlir::Value> operands;
      for (auto oper : op->getOperands()) {
        // TODO: catch failures here.
        operands.push_back(mapping.lookup(oper));
      }

      llvm::SmallVector<mlir::Value> results;
      if (mlir::failed(
              op.lowerBody(rewriter, globals, operands, results, haveMore))) {
        return mlir::failure();
      }

      // Map results
      mapping.map(op->getResults(), results);
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
  llvm::SmallVector<PipelineOp> pipelineOps(
      getOperation().getOps<PipelineOp>());
  mlir::IRRewriter rewriter(getOperation());
  rewriter.setInsertionPointToEnd(getOperation().getBody());

  bool hadFailure = false;
  for (auto op : pipelineOps) {
    if (mlir::failed(lowerPipeline(rewriter, op))) {
      hadFailure = true;
    }
  }

  if (hadFailure) {
    return signalPassFailure();
  }
}

} // namespace columnar