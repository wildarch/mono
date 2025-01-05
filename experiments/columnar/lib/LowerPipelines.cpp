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

template <typename T> class OpConversion : public mlir::OpConversionPattern<T> {
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op,
                  typename mlir::OpConversionPattern<T>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace

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
  auto &body = chunkOp.getBody().emplaceBlock();
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
  auto &body = chunkOp.getBody().emplaceBlock();
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
  auto &body = chunkOp.getBody().emplaceBlock();
  input = body.addArgument(input.getType(), input.getLoc());
  sel = body.addArgument(sel.getType(), sel.getLoc());
  rewriter.setInsertionPointToStart(&body);

  rewriter.create<TensorPrintOp>(op.getLoc(), op.getName(), input, sel);
  rewriter.create<ChunkYieldOp>(op.getLoc(), mlir::ValueRange{});

  rewriter.replaceOp(op, chunkOp);
  return mlir::success();
  return mlir::success();
}

void LowerPipelines::runOnOperation() {
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
}

} // namespace columnar