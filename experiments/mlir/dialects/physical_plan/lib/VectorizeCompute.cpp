#include "PhysicalPlanOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

#include "PhysicalPlanPasses.h"
#include "llvm/ADT/ArrayRef.h"
#include <llvm-17/llvm/Support/Casting.h>

namespace physicalplan {

#define GEN_PASS_DEF_VECTORIZECOMPUTE
#include "PhysicalPlanPasses.h.inc"

namespace {

class VectorizeCompute : public impl::VectorizeComputeBase<VectorizeCompute> {
public:
  using impl::VectorizeComputeBase<VectorizeCompute>::VectorizeComputeBase;

  void runOnOperation() final;
};

class ComputeOpConversion : public mlir::OpConversionPattern<ComputeOp> {
  using mlir::OpConversionPattern<ComputeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ComputeReturnOpConversion
    : public mlir::OpConversionPattern<ComputeReturnOp> {
  using mlir::OpConversionPattern<ComputeReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class AddIOpConversion : public mlir::OpConversionPattern<mlir::arith::AddIOp> {
  using mlir::OpConversionPattern<mlir::arith::AddIOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

template <typename SourceOp>
class ArithOpConversion : public mlir::ConversionPattern {
public:
  ArithOpConversion(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(typeConverter, SourceOp::getOperationName(),
                                /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace

static constexpr std::array<std::int64_t, 1> VECTOR_SHAPE{
    8,
};

static bool areVectors(mlir::TypeRange types) {
  return llvm::all_of(
      types, [](mlir::Type type) { return llvm::isa<mlir::VectorType>(type); });
}

static bool blockArgsAreVectors(ComputeOp op) {
  return areVectors(op.getBody().getArgumentTypes());
}

static bool operandsAreVectors(mlir::Operation *op) {
  return areVectors(op->getOperandTypes());
}

static std::optional<mlir::Type> scalarToVectors(mlir::Type type) {
  if (!type.isIntOrFloat()) {
    return std::nullopt;
  }
  return mlir::VectorType::get(VECTOR_SHAPE, type);
}

static std::optional<mlir::Type> blockToBlock(BlockType type) { return type; }

// ============================================================================
// ============================ matchAndRewrite ===============================
// ============================================================================

mlir::LogicalResult ComputeOpConversion::matchAndRewrite(
    ComputeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (blockArgsAreVectors(op)) {
    // Already converted.
    return mlir::failure();
  }

  auto result = mlir::failure();
  rewriter.updateRootInPlace(op, [&]() {
    result = rewriter.convertRegionTypes(&op.getBody(), *typeConverter);
  });

  if (mlir::failed(result)) {
    return op->emitError("failed to convert region types");
  }

  return result;
}

mlir::LogicalResult AddIOpConversion::matchAndRewrite(
    mlir::arith::AddIOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getLhs(),
                                                   adaptor.getRhs());
  return mlir::success();
}

template <typename T>
mlir::LogicalResult ArithOpConversion<T>::matchAndRewrite(
    mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.updateRootInPlace(op, [&]() {
    // Replace operands
    op->setOperands(operands);

    // Set result types
    for (auto res : op->getResults()) {
      res.setType(typeConverter->convertType(res.getType()));
    }
  });

  return mlir::success();
}

mlir::LogicalResult ComputeReturnOpConversion::matchAndRewrite(
    ComputeReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<ComputeReturnOp>(op, adaptor.getInput());
  return mlir::success();
}

void VectorizeCompute::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addDynamicallyLegalOp<ComputeOp>(blockArgsAreVectors);
  target.addDynamicallyLegalOp<mlir::arith::AddIOp, ComputeReturnOp>(
      operandsAreVectors);

  mlir::TypeConverter typeConverter;
  typeConverter.addConversion(scalarToVectors);
  typeConverter.addConversion(blockToBlock);

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ComputeOpConversion, ComputeReturnOpConversion,
               ArithOpConversion<mlir::arith::AddIOp>,
               ArithOpConversion<mlir::arith::CmpIOp>>(typeConverter,
                                                       &getContext());

  // Top-level ops.

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan