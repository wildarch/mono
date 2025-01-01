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

static constexpr std::size_t BLOCK_SIZE = 1024;

static mlir::Value buildIdentitySelectionVector(mlir::Location loc,
                                                mlir::OpBuilder &builder) {
  auto blockSizeOp = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIndexAttr(BLOCK_SIZE));

  auto resultType = mlir::RankedTensorType::get(
      llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic},
      builder.getIndexType());
  auto tensorOp = builder.create<mlir::tensor::GenerateOp>(
      loc, resultType, mlir::ValueRange{blockSizeOp},
      [](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
        builder.create<mlir::tensor::YieldOp>(loc, args[0]);
      });

  return tensorOp;
}

template <>
mlir::LogicalResult OpConversion<ConstantOp>::matchAndRewrite(
    ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (llvm::isa<SelIdAttr>(op.getValue())) {
    // Make identity selection vector.
    auto selOp = buildIdentitySelectionVector(op.getLoc(), rewriter);
    rewriter.replaceOp(op, selOp);
    return mlir::success();
  }

  return mlir::failure();
}

template <>
mlir::LogicalResult OpConversion<ReadTableOp>::matchAndRewrite(
    ReadTableOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<TensorReadColumnOp>(
      op, typeConverter->convertType(op.getType()), op.getTable(),
      op.getColumn());
  return mlir::success();
}

template <>
mlir::LogicalResult OpConversion<PrintOp>::matchAndRewrite(
    PrintOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<TensorPrintOp>(
      op, op.getName(), adaptor.getInput(), adaptor.getSel());
  return mlir::success();
}

void LowerPipelines::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<ColumnarDialect>();
  target.addLegalOp<PipelineOp>();
  target.addLegalOp<TensorReadColumnOp>();
  target.addLegalOp<TensorPrintOp>();
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
  patterns.add<OpConversion<ConstantOp>, OpConversion<ReadTableOp>,
               OpConversion<PrintOp>>(typeConverter, &getContext());

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar