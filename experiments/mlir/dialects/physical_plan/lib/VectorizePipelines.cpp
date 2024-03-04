#include <array>
#include <llvm-17/llvm/ADT/SmallVector.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanOps.h"
#include "PhysicalPlanPasses.h"
#include "PhysicalPlanTypes.h"

namespace physicalplan {

#define GEN_PASS_DEF_VECTORIZEPIPELINES
#include "PhysicalPlanPasses.h.inc"

static constexpr std::array<std::int64_t, 1> VECTOR_SHAPE{
    8,
};

namespace {

class VectorizePipelines
    : public impl::VectorizePipelinesBase<VectorizePipelines> {
public:
  using impl::VectorizePipelinesBase<
      VectorizePipelines>::VectorizePipelinesBase;

  void runOnOperation() final;
};

class ScanOpConversion : public mlir::OpConversionPattern<ScanOp> {
  using mlir::OpConversionPattern<ScanOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ScanOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ComputeOpConversion : public mlir::OpConversionPattern<ComputeOp> {
  using mlir::OpConversionPattern<ComputeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class WriteArrayOpConversion : public mlir::OpConversionPattern<WriteArrayOp> {
  using mlir::OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(WriteArrayOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace

static mlir::FailureOr<mlir::ValueRange> getInputVectors(mlir::Value input) {
  if (auto packOp = input.getDefiningOp<PackVectorsOp>()) {
    return packOp.getInputs();
  } else if (auto castOp =
                 input.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    return castOp.getInputs();
  }

  return mlir::failure();
}

static mlir::Value replaceWithVectors(mlir::Operation *origOp,
                                      mlir::Operation *vecOp,
                                      mlir::PatternRewriter &builder) {
  return builder.replaceOpWithNewOp<PackVectorsOp>(origOp, vecOp->getResults());
}

mlir::LogicalResult ScanOpConversion::matchAndRewrite(
    ScanOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto blockType =
      llvm::cast<BlockType>(op.getBody().front().getArgument(0).getType());

  mlir::TypeConverter::SignatureConversion sigConv(/*numOrigInputs=*/1);
  llvm::SmallVector<mlir::Type> vectorTypes;
  if (mlir::failed(typeConverter->convertTypes(mlir::TypeRange{blockType},
                                               vectorTypes))) {
    return mlir::failure();
  }
  sigConv.addInputs(0, vectorTypes);

  auto vecOp = rewriter.create<VectorizedScanOp>(
      op->getLoc(), op.getNumberOfTuples(), op.getColumnPointers());

  // Clone the body and convert the types.
  rewriter.cloneRegionBefore(op.getBody(), vecOp.getBody(),
                             vecOp.getBody().end());
  if (mlir::failed(rewriter.convertRegionTypes(&vecOp.getBody(), *typeConverter,
                                               &sigConv))) {
    return mlir::failure();
  }

  rewriter.replaceOp(op, vecOp);
  return mlir::success();
}

mlir::LogicalResult ComputeOpConversion::matchAndRewrite(
    ComputeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto inputs = getInputVectors(adaptor.getInput());
  if (mlir::failed(inputs)) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type> resultTypes;
  if (mlir::failed(typeConverter->convertTypes(mlir::TypeRange{op.getType()},
                                               resultTypes))) {
    return mlir::failure();
  }

  auto vecOp = rewriter.create<VectorizeOp>(op->getLoc(), resultTypes, *inputs);
  rewriter.inlineRegionBefore(op.getBody(), vecOp.getBody(),
                              vecOp.getBody().end());
  replaceWithVectors(op, vecOp, rewriter);
  return mlir::success();
}

mlir::LogicalResult WriteArrayOpConversion::matchAndRewrite(
    WriteArrayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto inputs = getInputVectors(adaptor.getInput());
  if (mlir::failed(inputs)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<VectorizedWriteArrayOp>(
      op, *inputs, op.getOutputColumnPointers(), op.getOffsetPointer(),
      op.getCapacity());
  return mlir::success();
}

static mlir::LogicalResult
convertComputeReturn(ComputeReturnOp op, mlir::PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<VectorizeReturnOp>(op, op.getInput());
  return mlir::success();
}

static std::optional<mlir::LogicalResult>
blockToVectors(BlockType blockType,
               llvm::SmallVectorImpl<mlir::Type> &outputTypes) {
  for (auto colType : blockType.getTypes()) {
    outputTypes.emplace_back(mlir::VectorType::get(VECTOR_SHAPE, colType));
  }

  return mlir::success();
}

static mlir::LogicalResult erasePack(PackVectorsOp op,
                                     mlir::PatternRewriter &rewriter) {
  if (!op->use_empty()) {
    return mlir::failure();
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

void VectorizePipelines::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();

  //  Vectorized ops are allowed.
  target.addLegalOp<VectorizedScanOp>();
  target.addLegalOp<VectorizedWriteArrayOp>();
  target.addLegalOp<VectorizeOp>();
  target.addLegalOp<VectorizeReturnOp>();

  mlir::TypeConverter blockConverter;
  blockConverter.addConversion(blockToVectors);

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ScanOpConversion>(blockConverter, &getContext());
  patterns.add<WriteArrayOpConversion>(blockConverter, &getContext());
  patterns.add<ComputeOpConversion>(blockConverter, &getContext());
  patterns.add(convertComputeReturn);
  patterns.add(erasePack);

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan