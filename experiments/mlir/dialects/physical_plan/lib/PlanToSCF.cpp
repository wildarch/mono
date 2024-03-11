#include <array>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

static mlir::LogicalResult scanLowering(VectorizedScanOp op,
                                        mlir::PatternRewriter &rewriter) {
  const std::int64_t VECTOR_SIZE = 8;

  // Declare the memrefs for the columns.
  llvm::SmallVector<mlir::Value> columns;
  for (auto [ptr, argTy] : llvm::zip_equal(op.getColumnPointers(),
                                           op.getBody().getArgumentTypes())) {
    auto vecTy = llvm::cast<mlir::VectorType>(argTy);
    assert(vecTy.getShape().front() == VECTOR_SIZE);
    auto memRefTy = mlir::MemRefType::get(
        std::array<std::int64_t, 1>{std::int64_t(op.getNumberOfTuples())},
        vecTy.getElementType());
    auto ptrOp = rewriter.create<mlir::arith::ConstantIntOp>(
        op->getLoc(), ptr, rewriter.getI64Type());
    columns.emplace_back(
        rewriter.create<DeclMemRefOp>(op->getLoc(), memRefTy, ptrOp));
  }

  // Loop over the input
  auto zeroOp = rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 0);
  auto nrofTuplesOp = rewriter.create<mlir::arith::ConstantIndexOp>(
      op->getLoc(), op.getNumberOfTuples());
  auto stepOp =
      rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), VECTOR_SIZE);
  auto forOp = rewriter.create<mlir::scf::ForOp>(op->getLoc(), zeroOp,
                                                 nrofTuplesOp, stepOp);
  auto &forBlock = forOp.getLoopBody().front();
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&forBlock);

    // Erase dummy yield op
    rewriter.eraseOp(&forBlock.back());

    // Load the vectors
    llvm::SmallVector<mlir::vector::LoadOp> loadOps;
    for (auto column : columns) {
      auto columnType = llvm::cast<mlir::MemRefType>(column.getType());
      auto vecType =
          mlir::VectorType::get(std::array<std::int64_t, 1>{VECTOR_SIZE},
                                columnType.getElementType());
      loadOps.emplace_back(rewriter.create<mlir::vector::LoadOp>(
          op->getLoc(), vecType, column,
          mlir::ValueRange{forOp.getInductionVar()}));
    }

    // Inline the scan body.
    mlir::IRMapping mapping;
    for (auto [arg, loadOp] :
         llvm::zip_equal(op.getBody().getArguments(), loadOps)) {
      assert(arg.getType() == loadOp.getType());
      mapping.map(arg, loadOp);
    }

    for (auto &vecOp : op.getBody().front()) {
      auto *newOp = rewriter.clone(vecOp, mapping);
      mapping.map(&vecOp, newOp);
    }
  }

  // forOp replaces the old op.
  rewriter.eraseOp(op);

  return mlir::success();
}

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

  // rewriter.replaceOpWithNewOp<VectorizedScanReturnOp>(op);
  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op);
  return mlir::success();
}

void PlanToSCF::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addIllegalDialect<PhysicalPlanDialect>();
  // Low-level ops are allowed.
  target.addLegalOp<DeclMemRefOp>();
  target.addLegalOp<ClaimSliceOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(writeArrayLowering);
  patterns.add(scanLowering);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan