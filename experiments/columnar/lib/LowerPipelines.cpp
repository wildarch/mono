#include "columnar/Columnar.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  auto scannerOp = builder.create<TableScannerOpenOp>(getLoc(), getTable());
  newGlobals.push_back(scannerOp);

  // Open columns
  for (auto col : getColumnsToRead()) {
    auto columnOp = builder.create<TableColumnOpenOp>(getLoc(), col);
    newGlobals.push_back(columnOp);
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
  auto claimOp = builder.create<TableScannerClaimChunkOp>(getLoc(), scanner);

  // If the chunk has size > 0, there may be more to read.
  auto zeroOp = builder.create<mlir::arith::ConstantOp>(
      getLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  auto haveRowsOp = builder.create<mlir::arith::CmpIOp>(
      getLoc(), mlir::arith::CmpIPredicate::ugt, claimOp.getSize(), zeroOp);
  haveMore.push_back(haveRowsOp);

  auto selOp = buildIotaSelectionVector(builder, getLoc(), claimOp.getSize());
  results.push_back(selOp);

  // Read the columns
  for (auto [col, type] : llvm::zip_equal(columns, getCol().getTypes())) {
    auto tensorType = convertType(llvm::cast<ColumnType>(type));
    if (!tensorType) {
      return mlir::failure();
    }

    auto readOp = builder.create<TableColumnReadOp>(
        getLoc(), tensorType, col, claimOp.getStart(), claimOp.getSize());
    results.push_back(readOp);
  }

  return mlir::success();
}

// PrintOp
mlir::LogicalResult
PrintOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                         llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  auto handle = builder.create<PrintOpenOp>(getLoc());
  newGlobals.push_back(handle);
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
  auto chunk = builder.create<PrintChunkAllocOp>(getLoc(), nrows);

  // Append columns
  for (auto input : adaptor.getInputs()) {
    builder.create<PrintChunkAppendOp>(getLoc(), chunk, input, sel);
  }

  // Write chunk
  builder.create<PrintWriteOp>(getLoc(), handle, chunk);
  return mlir::success();
}

static void addArgumentsFor(mlir::Block &block, mlir::ValueRange values) {
  for (auto v : values) {
    block.addArgument(v.getType(), v.getLoc());
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

    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(), globals);

    // Globals are available in all blocks
    addArgumentsFor(bodyBlock, globals);
    addArgumentsFor(globalCloseBlock, globals);
  }

  // Global free
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&globalCloseBlock);

    auto args = globalCloseBlock.getArguments();
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

    auto args = bodyBlock.getArguments();
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