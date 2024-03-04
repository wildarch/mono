#include <array>
#include <llvm-17/llvm/Support/Casting.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanOps.h"
#include "PhysicalPlanPasses.h"
#include "PhysicalPlanTypes.h"
#include "llvm/ADT/STLExtras.h"

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

class AddIOpConversion : public mlir::OpConversionPattern<mlir::arith::AddIOp> {
  using mlir::OpConversionPattern<mlir::arith::AddIOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ComputeReturnOpConversion
    : public mlir::OpConversionPattern<ComputeReturnOp> {
  using mlir::OpConversionPattern<ComputeReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeReturnOp op, OpAdaptor adaptor,
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
  if (mlir::failed(
          rewriter.convertRegionTypes(&vecOp.getBody(), *typeConverter))) {
    return mlir::failure();
  }

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

mlir::LogicalResult AddIOpConversion::matchAndRewrite(
    mlir::arith::AddIOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
  return mlir::success();
}

mlir::LogicalResult ComputeReturnOpConversion::matchAndRewrite(
    ComputeReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<VectorizeReturnOp>(op, adaptor.getInput());
  return mlir::success();
}

static mlir::LogicalResult inlineVectorizeOp(VectorizeOp op,
                                             mlir::PatternRewriter &rewriter) {
  if (!llvm::isa_and_nonnull<VectorizeReturnOp>(
          op.getBody().front().getTerminator())) {
    // Need to convert return op first.
    return mlir::failure();
  }

  // Inputs are propagated.
  llvm::SmallVector<mlir::Value> results(op.getInputs());

  // Inline the body
  mlir::IRMapping mapping;
  for (auto [arg, input] :
       llvm::zip_equal(op.getBody().getArguments(), op.getInputs())) {
    mapping.map(arg, input);
  }
  bool foundReturnOp = false;
  for (auto &op : op.getBody().front()) {
    if (auto retOp = llvm::dyn_cast<VectorizeReturnOp>(op)) {
      results.emplace_back(mapping.lookup(retOp.getInput()));
      foundReturnOp = true;
      break;
    }

    auto newOp = rewriter.clone(op, mapping);
    mapping.map(&op, newOp);
  }

  if (!foundReturnOp) {
    return op->emitError("Could not find return op");
  }

  llvm::SmallVector<mlir::Type> resultTypes;
  for (auto r : results) {
    resultTypes.emplace_back(r.getType());
  }

  // Sanity check.
  if (!llvm::equal(resultTypes, op->getResultTypes())) {
    return op->emitError("Op results incompatible with replacement values: ")
           << op.getResultTypes() << " vs. " << results;
  }

  rewriter.replaceOp(op, results);
  return mlir::success();
}

bool hasVectorOperands(mlir::Operation *op) {
  return llvm::all_of(op->getOperandTypes(), [](mlir::Type t) {
    return llvm::isa<mlir::VectorType>(t);
  });
}

static std::optional<mlir::LogicalResult>
blockToVectors(BlockType blockType,
               llvm::SmallVectorImpl<mlir::Type> &outputTypes) {
  for (auto colType : blockType.getTypes()) {
    outputTypes.emplace_back(mlir::VectorType::get(VECTOR_SHAPE, colType));
  }

  return mlir::success();
}

static mlir::Type scalarToVectors(mlir::IntegerType type) {
  return mlir::VectorType::get(VECTOR_SHAPE, type);
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
  // target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addDynamicallyLegalOp<mlir::arith::AddIOp>(hasVectorOperands);

  //  Vectorized ops are allowed.
  target.addLegalOp<VectorizedScanOp>();
  target.addLegalOp<VectorizedWriteArrayOp>();
  // target.addLegalOp<VectorizeOp>();
  // target.addLegalOp<VectorizeReturnOp>();

  mlir::TypeConverter blockConverter;
  blockConverter.addConversion(blockToVectors);
  blockConverter.addConversion(scalarToVectors);

  mlir::RewritePatternSet patterns(&getContext());

  // Top-level ops.
  patterns.add<ScanOpConversion>(blockConverter, &getContext());
  patterns.add<WriteArrayOpConversion>(blockConverter, &getContext());
  patterns.add<ComputeOpConversion>(blockConverter, &getContext());
  patterns.add(inlineVectorizeOp);

  // Converts scalar ops.
  patterns.add<AddIOpConversion>(blockConverter, &getContext());
  patterns.add<ComputeReturnOpConversion>(blockConverter, &getContext());

  // Cleanup
  patterns.add(erasePack);

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan