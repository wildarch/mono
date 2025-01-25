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

/*
template <typename T> class OpConversion : public mlir::OpConversionPattern<T> {
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op,
                  typename mlir::OpConversionPattern<T>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};
*/

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

// SelTableOp
mlir::LogicalResult
SelTableOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                            llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  // Open a scanner
  auto scannerOp = builder.create<SelScannerOp>(getLoc(), getTable());
  newGlobals.push_back(scannerOp);
  return mlir::success();
}

mlir::LogicalResult SelTableOp::lowerGlobalClose(mlir::OpBuilder &builder,
                                                 mlir::ValueRange globals) {
  // TODO: close the scanner
  return mlir::success();
}

mlir::LogicalResult
SelTableOp::lowerBody(mlir::OpBuilder &builder, mlir::ValueRange globals,
                      mlir::ValueRange operands,
                      llvm::SmallVectorImpl<mlir::Value> &results) {
  auto scanner = globals[0];
  auto readOp = builder.create<TensorReadColumnOp>(
      getLoc(), convertType(getType()), scanner);
  results.push_back(readOp);
  return mlir::success();
}

// ReadTableOp
mlir::LogicalResult
ReadTableOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                             llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  // Open a scanner
  auto scannerOp = builder.create<OpenColumnOp>(getLoc(), getColumn());
  newGlobals.push_back(scannerOp);
  return mlir::success();
}

mlir::LogicalResult ReadTableOp::lowerGlobalClose(mlir::OpBuilder &builder,
                                                  mlir::ValueRange globals) {
  // TODO: close the scanner
  return mlir::success();
}

mlir::LogicalResult
ReadTableOp::lowerBody(mlir::OpBuilder &builder, mlir::ValueRange globals,
                       mlir::ValueRange operands,
                       llvm::SmallVectorImpl<mlir::Value> &results) {
  auto scanner = globals[0];
  auto readOp = builder.create<TensorReadColumnOp>(
      getLoc(), convertType(getType()), scanner);
  results.push_back(readOp);
  return mlir::success();
}

// PrintOp
mlir::LogicalResult
PrintOp::lowerGlobalOpen(mlir::OpBuilder &builder,
                         llvm::SmallVectorImpl<mlir::Value> &newGlobals) {
  // TODO: Open result printer
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
                   llvm::SmallVectorImpl<mlir::Value> &results) {
  Adaptor adaptor(operands, *this);

  auto sel = adaptor.getSel();
  for (auto [name, input] : llvm::zip_equal(getNames(), adaptor.getInputs())) {
    builder.create<TensorPrintOp>(getLoc(), name, input, sel);
  }

  return mlir::success();
}

/*
template <>
mlir::LogicalResult OpConversion<ReadTableOp>::matchAndRewrite(
    ReadTableOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultType = typeConverter->convertType(op.getType());
  // Open a scanner
  auto scannerOp = rewriter.create<OpenColumnOp>(op.getLoc(), op.getColumn());

  // Chunked read.
  auto chunkOp = rewriter.create<ChunkOp>(
      op.getLoc(), mlir::TypeRange{resultType}, mlir::ValueRange{});
  auto &body = chunkOp.getBody().front();
  rewriter.setInsertionPointToStart(&body);

  auto readOp =
      rewriter.create<TensorReadColumnOp>(op.getLoc(), resultType, scannerOp);
  rewriter.create<ChunkYieldOp>(op.getLoc(), mlir::ValueRange{readOp});

  rewriter.replaceOp(op, chunkOp);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<SelTableOp>::matchAndRewrite(
    SelTableOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultType = typeConverter->convertType(op.getType());
  // Open a scanner
  auto scannerOp = rewriter.create<SelScannerOp>(op.getLoc(), op.getTable());

  // Chunked read.
  auto chunkOp = rewriter.create<ChunkOp>(
      op.getLoc(), mlir::TypeRange{resultType}, mlir::ValueRange{});
  auto &body = chunkOp.getBody().front();
  rewriter.setInsertionPointToStart(&body);

  auto readOp =
      rewriter.create<TensorReadColumnOp>(op.getLoc(), resultType, scannerOp);
  rewriter.create<ChunkYieldOp>(op.getLoc(), mlir::ValueRange{readOp});

  rewriter.replaceOp(op, chunkOp);
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<PrintOp>::matchAndRewrite(
    PrintOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // TODO: Chunked read.
  auto input = adaptor.getInput();
  auto sel = adaptor.getSel();
  auto chunkOp = rewriter.create<ChunkOp>(op.getLoc(), mlir::TypeRange{},
                                          mlir::ValueRange{input, sel});
  auto &body = chunkOp.getBody().front();
  input = body.getArgument(0);
  sel = body.getArgument(1);
  rewriter.setInsertionPointToStart(&body);

  rewriter.create<TensorPrintOp>(op.getLoc(), op.getName(), input, sel);
  rewriter.create<ChunkYieldOp>(op.getLoc(), mlir::ValueRange{});

  rewriter.replaceOp(op, chunkOp);
  return mlir::success();
  return mlir::success();
}
*/

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
      if (mlir::failed(op.lowerBody(rewriter, globals, operands, results))) {
        return mlir::failure();
      }

      // Map results
      mapping.map(op->getResults(), results);
    }

    rewriter.create<PipelineLowYieldOp>(pipelineOp.getLoc(),
                                        mlir::ValueRange{});
  }

  rewriter.replaceOp(pipelineOp, lowerOp);
  return mlir::success();
}

void LowerPipelines::runOnOperation() {
  /*
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<ColumnarDialect>();
  target.addLegalOp<PipelineOp, ChunkOp, ChunkYieldOp, OpenColumnOp,
                    SelScannerOp, TensorReadColumnOp, TensorPrintOp>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();

  mlir::TypeConverter typeConverter;
  typeConverter.addConversion([&](ColumnType t) {
    return mlir::RankedTensorType::get(
        llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic},
        typeConverter.convertType(t.getElementType()));
  });

  // Element types
  typeConverter.addConversion(
      [](SelectType t) { return mlir::IndexType::get(t.getContext()); });
  typeConverter.addConversion([](mlir::FloatType t) { return t; });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<OpConversion<ReadTableOp>, OpConversion<PrintOp>,
               OpConversion<SelTableOp>>(typeConverter, &getContext());

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    return signalPassFailure();
  }
  */

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