#include <memory>

#include "MiniDialect.h"
#include "MiniOps.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace {
struct MiniLoweringPass
    : public mlir::PassWrapper<MiniLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MiniLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  }
  void runOnOperation() final;
};

struct FuncOpLowering
    : public mlir::OpConversionPattern<experiments_mlir::mini::FuncOp> {
  using OpConversionPattern<
      experiments_mlir::mini::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(experiments_mlir::mini::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Create a new non-mini function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConstantOpLowering
    : public mlir::OpConversionPattern<experiments_mlir::mini::ConstantOp> {
  using OpConversionPattern<
      experiments_mlir::mini::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(experiments_mlir::mini::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, op.getValue(),
                                                            32);
    return mlir::success();
  }
};

struct ReturnOpLowering
    : public mlir::OpRewritePattern<experiments_mlir::mini::ReturnOp> {
  using OpRewritePattern<experiments_mlir::mini::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(experiments_mlir::mini::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const final {
    // We lower "experiments_mlir::mini.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op->getOperands());
    return mlir::success();
  }
};

struct MiniToLLVMLoweringPass
    : public mlir::PassWrapper<MiniToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MiniToLLVMLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};

} // namespace

void MiniLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::BuiltinDialect, mlir::arith::ArithDialect,
                         mlir::func::FuncDialect>();
  target.addIllegalDialect<experiments_mlir::mini::MiniDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering, ConstantOpLowering, ReturnOpLowering>(
      &getContext());

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    signalPassFailure();
  }
}

void MiniToLLVMLoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (mlir::failed(
          mlir::applyFullConversion(module, target, std::move(patterns)))) {

    signalPassFailure();
  }
}

namespace experiments_mlir::mini {

std::unique_ptr<mlir::Pass> createMiniLoweringPass() {
  return std::make_unique<MiniLoweringPass>();
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<MiniToLLVMLoweringPass>();
}

} // namespace experiments_mlir::mini