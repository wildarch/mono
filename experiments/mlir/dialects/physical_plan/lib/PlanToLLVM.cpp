#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanOps.h"
#include "PhysicalPlanPasses.h"
#include "PhysicalPlanTypes.h"
#include <array>

namespace physicalplan {

#define GEN_PASS_DEF_PLANTOLLVM
#include "PhysicalPlanPasses.h.inc"

namespace {

class PlanToLLVM : public impl::PlanToLLVMBase<PlanToLLVM> {
public:
  using impl::PlanToLLVMBase<PlanToLLVM>::PlanToLLVMBase;

  void runOnOperation() final;
};

class DeclMemRefOpConversion : public mlir::OpConversionPattern<DeclMemRefOp> {
  using mlir::OpConversionPattern<DeclMemRefOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DeclMemRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ClaimSliceOpConversion : public mlir::OpConversionPattern<ClaimSliceOp> {
  using mlir::OpConversionPattern<ClaimSliceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ClaimSliceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace

mlir::LogicalResult DeclMemRefOpConversion::matchAndRewrite(
    DeclMemRefOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // conversion is of form:
  // memref<? x f32> -> !llvm.struct<(
  //                        ptr,              (allocated ptr = base ptr)
  //                        ptr,              (aligned ptr = base ptr)
  //                        i64,              (offset = 0)
  //                        array<1 x i64>,   (shape = [?])
  //                        array<1 x i64>)>  (stride = [1])
  auto offsetOp = rewriter.create<mlir::LLVM::ConstantOp>(
      op->getLoc(), rewriter.getI64Type(), 0);
  auto shape = rewriter.getDenseI64ArrayAttr(op.getType().getShape());
  auto shapeOp = rewriter.create<mlir::LLVM::ConstantOp>(
      op->getLoc(),
      rewriter.getType<mlir::LLVM::LLVMArrayType>(rewriter.getI64Type(), 1),
      shape);
  auto stride = rewriter.getDenseI64ArrayAttr(std::array<std::int64_t, 1>{1});
  auto strideOp = rewriter.create<mlir::LLVM::ConstantOp>(
      op->getLoc(),
      rewriter.getType<mlir::LLVM::LLVMArrayType>(rewriter.getI64Type(), 1),
      stride);
  auto structType = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, structType);
  // TODO: actually populate
  return mlir::success();
}

mlir::LogicalResult ClaimSliceOpConversion::matchAndRewrite(
    ClaimSliceOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  return op->emitError("not implemented");
}

void PlanToLLVM::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<DeclMemRefOpConversion>(typeConverter, &getContext());
  patterns.add<ClaimSliceOpConversion>(typeConverter, &getContext());

  if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace physicalplan