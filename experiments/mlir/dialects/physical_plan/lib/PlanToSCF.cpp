#include <array>

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

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanOps.h"
#include "PhysicalPlanPasses.h"
#include "PhysicalPlanTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace physicalplan {

#define GEN_PASS_DEF_PLANTOSCF
#include "PhysicalPlanPasses.h.inc"

namespace {

class PlanToSCF : public impl::PlanToSCFBase<PlanToSCF> {
public:
  using impl::PlanToSCFBase<PlanToSCF>::PlanToSCFBase;

  void runOnOperation() final;
};

} // namespace

static mlir::LogicalResult writeArrayLowering(VectorizedWriteArrayOp op,
                                              mlir::PatternRewriter &rewriter) {
  for (auto [i, input] : llvm::enumerate(op.getInputs())) {
    auto inputType = llvm::cast<mlir::VectorType>(input.getType());
    auto claimSize = rewriter.create<mlir::arith::ConstantIntOp>(
        op->getLoc(), inputType.getShape()[0], rewriter.getI64Type());
    auto offsetPtr = rewriter.create<mlir::arith::ConstantIntOp>(
        op->getLoc(), op.getOffsetPointer(), rewriter.getI64Type());
    auto capacity = rewriter.create<mlir::arith::ConstantIntOp>(
        op->getLoc(), op.getCapacity(), rewriter.getI64Type());
    auto claimOp = rewriter.create<ClaimSliceOp>(
        op->getLoc(), rewriter.getIndexType(), rewriter.getI1Type(), claimSize,
        offsetPtr, capacity);
    // TODO: check error status

    // Create base memref
    auto basePtr = rewriter.create<mlir::arith::ConstantIntOp>(
        op->getLoc(), op.getOutputColumnPointers()[i], rewriter.getI64Type());
    auto baseType = mlir::MemRefType::get(
        std::array<std::int64_t, 1>{mlir::ShapedType::kDynamic},
        inputType.getElementType());
    mlir::Value base =
        rewriter.create<DeclMemRefOp>(op.getLoc(), baseType, basePtr);

    // Make the store happen.
    rewriter.create<mlir::vector::StoreOp>(
        op->getLoc(), input, base, mlir::ValueRange{claimOp.getOffset()});
  }

  rewriter.replaceOpWithNewOp<VectorizedScanReturnOp>(op);
  return mlir::success();
}

void PlanToSCF::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addIllegalDialect<PhysicalPlanDialect>();
  // Low-level ops are allowed.
  target.addLegalOp<DeclMemRefOp>();
  target.addLegalOp<ClaimSliceOp>();
  // Temp
  target.addLegalOp<VectorizedScanOp>();
  target.addLegalOp<VectorizedScanReturnOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(writeArrayLowering);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan