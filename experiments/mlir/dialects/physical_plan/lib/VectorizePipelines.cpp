#include <array>
#include <llvm-17/llvm/Support/Casting.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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

class FilterOpConversion : public mlir::OpConversionPattern<FilterOp> {
  using mlir::OpConversionPattern<FilterOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FilterOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class WriteArrayOpConversion : public mlir::OpConversionPattern<WriteArrayOp> {
  using mlir::OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(WriteArrayOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

struct InputVectors {
  mlir::ValueRange inputs;
  mlir::Value mask;
};

} // namespace

static mlir::FailureOr<InputVectors>
getInputVectors(mlir::Value input, mlir::PatternRewriter &rewriter) {
  if (auto packOp = input.getDefiningOp<PackVectorsOp>()) {
    return InputVectors{packOp.getInputs(), packOp.getMask()};
  } else if (auto castOp =
                 input.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    return InputVectors{castOp.getInputs(), mlir::Value()};
  }

  return mlir::failure();
}

static mlir::Value replaceWithVectors(mlir::Operation *origOp,
                                      mlir::ValueRange vectors,
                                      mlir::Value mask,
                                      mlir::PatternRewriter &rewriter) {
  return rewriter.replaceOpWithNewOp<PackVectorsOp>(origOp, vectors, mask);
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

static mlir::LogicalResult
inlineBlock(mlir::Block &block, mlir::ValueRange inputs,
            mlir::ConversionPatternRewriter &rewriter,
            llvm::SmallVector<mlir::Value> &results) {
  // Map inputs to block arguments.
  mlir::IRMapping mapping;
  for (auto [arg, input] : llvm::zip_equal(block.getArguments(), inputs)) {
    if (arg.getType() != input.getType()) {
      return block.getParentOp()->emitError(
          "cannot inline because the block arguments do not "
          "match the input types");
    }
    mapping.map(arg, input);
  }

  // Clone the body ops.
  bool foundTerminator = false;
  auto terminator = block.getTerminator();
  for (auto &op : block) {
    if (&op == terminator) {
      for (auto oper : op.getOperands()) {
        results.emplace_back(mapping.lookup(oper));
      }

      foundTerminator = true;
      break;
    }

    auto newOp = rewriter.clone(op, mapping);
    mapping.map(&op, newOp);
  }

  if (!foundTerminator) {
    return block.getParentOp()->emitError("Could not find return op");
  }

  return mlir::success();
}

mlir::LogicalResult ComputeOpConversion::matchAndRewrite(
    ComputeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto inputs = getInputVectors(adaptor.getInput(), rewriter);
  if (mlir::failed(inputs)) {
    return mlir::failure();
  }

  // Inputs are propagated.
  llvm::SmallVector<mlir::Value> results(inputs->inputs);
  // Inline body ops.
  if (mlir::failed(inlineBlock(op.getBody().front(), inputs->inputs, rewriter,
                               results))) {
    return mlir::failure();
  }

  replaceWithVectors(op, results, inputs->mask, rewriter);
  return mlir::success();
}

mlir::LogicalResult FilterOpConversion::matchAndRewrite(
    FilterOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto inputs = getInputVectors(adaptor.getInput(), rewriter);
  if (mlir::failed(inputs)) {
    return mlir::failure();
  }

  // Merge masks
  assert(op.getColumn() < inputs->inputs.size());
  auto mask = inputs->inputs[op.getColumn()];
  if (inputs->mask) {
    mask =
        rewriter.create<mlir::arith::AndIOp>(op->getLoc(), inputs->mask, mask);
  }

  replaceWithVectors(op, inputs->inputs, mask, rewriter);
  return mlir::success();
}

mlir::LogicalResult WriteArrayOpConversion::matchAndRewrite(
    WriteArrayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto inputs = getInputVectors(adaptor.getInput(), rewriter);
  if (mlir::failed(inputs)) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> selectedInputs;
  for (auto col :
       op.getSelectedColumns().getAsValueRange<mlir::IntegerAttr>()) {
    auto idx = col.getZExtValue();
    assert(idx < inputs->inputs.size());
    selectedInputs.emplace_back(inputs->inputs[idx]);
  }

  rewriter.replaceOpWithNewOp<VectorizedWriteArrayOp>(
      op, selectedInputs, inputs->mask, op.getOutputColumnPointers(),
      op.getOffsetPointer(), op.getCapacity());
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
  target.addLegalDialect<mlir::vector::VectorDialect>();
  // Should have been converted in the Vectorize Compute pass.
  target.addLegalDialect<mlir::arith::ArithDialect>();

  //  Vectorized ops are allowed.
  target.addLegalOp<VectorizedScanOp>();
  target.addLegalOp<VectorizedWriteArrayOp>();

  mlir::TypeConverter blockConverter;
  blockConverter.addConversion(blockToVectors);

  mlir::RewritePatternSet patterns(&getContext());

  // Top-level ops.
  patterns.add<ScanOpConversion>(blockConverter, &getContext());
  patterns.add<WriteArrayOpConversion>(blockConverter, &getContext());
  patterns.add<ComputeOpConversion>(blockConverter, &getContext());
  patterns.add<FilterOpConversion>(blockConverter, &getContext());

  // Cleanup
  patterns.add(erasePack);

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan