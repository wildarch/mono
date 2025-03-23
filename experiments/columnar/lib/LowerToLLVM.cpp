#include "columnar/Columnar.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/FormatVariadic.h"

namespace columnar {

#define GEN_PASS_DEF_LOWERTOLLVM
#include "columnar/Passes.h.inc"

namespace {

class LowerToLLVM : public impl::LowerToLLVMBase<LowerToLLVM> {
public:
  using impl::LowerToLLVMBase<LowerToLLVM>::LowerToLLVMBase;

  void runOnOperation() final;
};

class AllocStructOpLowering
    : public mlir::ConvertOpToLLVMPattern<AllocStructOp> {
  using mlir::ConvertOpToLLVMPattern<AllocStructOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(AllocStructOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class GetStructElementOpLowering
    : public mlir::ConvertOpToLLVMPattern<GetStructElementOp> {
  using mlir::ConvertOpToLLVMPattern<
      GetStructElementOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(GetStructElementOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ConstantStringOpLowering
    : public mlir::ConvertOpToLLVMPattern<ConstantStringOp> {
  using mlir::ConvertOpToLLVMPattern<ConstantStringOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstantStringOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace

static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                           mlir::OpBuilder &builder,
                                           llvm::StringRef name,
                                           llvm::StringRef value,
                                           mlir::ModuleOp module) {
  // Create the global at the entry of the module.
  mlir::LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    std::string nullTerminated = value.str();
    nullTerminated += '\0';
    auto type = mlir::LLVM::LLVMArrayType::get(builder.getI8Type(),
                                               nullTerminated.size());
    global = builder.create<mlir::LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
        builder.getStringAttr(nullTerminated),
        /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  auto globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  auto cst0 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                     builder.getIndexAttr(0));
  return builder.create<mlir::LLVM::GEPOp>(
      loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
      global.getType(), globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
}

mlir::LogicalResult AllocStructOpLowering::matchAndRewrite(
    AllocStructOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // get or insert malloc
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto mallocOp = mlir::LLVM::lookupOrCreateMallocFn(module, getIndexType());

  auto structType = op.getResult().getType().getPointee();
  auto llvmStructType = getTypeConverter()->convertType(structType);
  if (!llvmStructType) {
    return op->emitError("cannot convert struct type ")
           << structType << " to LLVM";
  }

  // get size of the struct to allocate
  auto size = getSizeInBytes(op.getLoc(), llvmStructType, rewriter);
  if (!size) {
    return op->emitError("cannot determine size of type ") << llvmStructType;
  }

  // Call to malloc
  auto callOp = rewriter.create<mlir::LLVM::CallOp>(
      op.getLoc(), mallocOp.getFunctionType(), mallocOp.getSymNameAttr(), size);

  // Create the struct locally first
  mlir::Value structVal =
      rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), llvmStructType);
  for (auto [idx, value] : llvm::enumerate(adaptor.getValues())) {
    structVal = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), structVal, value, idx);
  }

  // Then write it to the heap-allocated memory
  rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), structVal,
                                       callOp->getResult(0));

  rewriter.replaceOp(op, callOp);
  return mlir::success();
}

mlir::LogicalResult GetStructElementOpLowering::matchAndRewrite(
    GetStructElementOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto structType = op.getValue().getType().getPointee();
  auto llvmStructType = getTypeConverter()->convertType(structType);
  if (!llvmStructType) {
    return op->emitError("cannot convert struct type ")
           << structType << " to LLVM";
  }

  // Load the struct
  auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), llvmStructType,
                                                    adaptor.getValue());
  // Extract the value
  rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(op, loadOp,
                                                          op.getIndex());
  return mlir::success();
}

mlir::LogicalResult ConstantStringOpLowering::matchAndRewrite(
    ConstantStringOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto value = op.getValue();
  auto newOp = getOrCreateGlobalString(op.getLoc(), rewriter, value, value,
                                       op->getParentOfType<mlir::ModuleOp>());
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

static mlir::LogicalResult findRuntimeCallsIn(
    mlir::Operation *op,
    llvm::DenseMap<mlir::StringAttr, mlir::FunctionType> &called) {
  bool hadError = false;
  op->walk([&](RuntimeCallOp op) {
    auto type = mlir::FunctionType::get(
        op->getContext(), op.getInputs().getTypes(), op->getResultTypes());
    auto exist = called.lookup(op.getFuncAttr());
    if (!exist) {
      called[op.getFuncAttr()] = type;
    } else if (type != exist) {
      op->emitError("runtime function '")
          << op.getFunc() << "' is used with different function types ("
          << exist << " vs. " << type << ")";
      hadError = true;
    }
  });

  return mlir::failure(hadError);
}

static mlir::SymbolRefAttr moveToFunction(mlir::Region &region,
                                          std::size_t pipeIdx,
                                          llvm::StringLiteral partName,
                                          mlir::IRRewriter &rewriter) {
  if (region.empty()) {
    return mlir::SymbolRefAttr();
  }

  auto name = llvm::formatv("pipe{}_{}", pipeIdx, partName).sstr<32>();
  auto &block = region.front();

  // Change terminator to ReturnOp appropriate for FuncOp.
  mlir::func::ReturnOp returnOp;
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto yieldOp = llvm::cast<PipelineLowYieldOp>(block.getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    returnOp = rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
        yieldOp, yieldOp.getValues());
  }

  // Inputs types match block arguments, and the function returns whatever the
  // block yields.
  auto funcType = rewriter.getFunctionType(block.getArgumentTypes(),
                                           returnOp->getOperandTypes());
  auto funcOp =
      rewriter.create<mlir::func::FuncOp>(region.getLoc(), name, funcType);
  rewriter.inlineRegionBefore(region, funcOp.getBody(),
                              funcOp.getBody().begin());

  return mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr());
}

static void pipelineMakeRef(PipelineLowOp op, std::size_t idx,
                            mlir::IRRewriter &rewriter) {
  auto globalOpenSym =
      moveToFunction(op.getGlobalOpen(), idx, "globalOpen", rewriter);
  auto bodySym = moveToFunction(op.getBody(), idx, "body", rewriter);
  auto globalCloseSym =
      moveToFunction(op.getGlobalClose(), idx, "globalClose", rewriter);

  rewriter.replaceOpWithNewOp<PipelineRefOp>(op, globalOpenSym, bodySym,
                                             globalCloseSym);
}

void LowerToLLVM::runOnOperation() {
  // Find called runtime functions
  llvm::DenseMap<mlir::StringAttr, mlir::FunctionType> called;
  if (mlir::failed(findRuntimeCallsIn(getOperation(), called))) {
    return signalPassFailure();
  }

  // Register as external functions
  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(getOperation().getBody());
  auto funcPrivateAttr = builder.getStringAttr("private");
  for (auto [name, type] : called) {
    builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), name, type, funcPrivateAttr,
        /*arg_attrs=*/nullptr, /*res_attrs*/ nullptr);
  }

  // Replace with function calls
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<RuntimeCallOp> callOps;
  getOperation()->walk([&](RuntimeCallOp op) { callOps.push_back(op); });
  for (auto op : callOps) {
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getFuncAttr(), op.getResultTypes(), op.getInputs());
  }

  // Replace pipeline blocks with references to functions.
  llvm::SmallVector<PipelineLowOp> pipelines(
      getOperation().getOps<PipelineLowOp>());
  for (auto [i, op] : llvm::enumerate(pipelines)) {
    rewriter.setInsertionPoint(op);
    pipelineMakeRef(op, i, rewriter);
  }

  // TODO: Invoke LLVM lowering to lower the functions.
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<PipelineRefOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());
  typeConverter.addConversion([](PointerType t) {
    return mlir::LLVM::LLVMPointerType::get(t.getContext());
  });
  typeConverter.addConversion([](ScannerHandleType t) {
    return mlir::LLVM::LLVMPointerType::get(t.getContext());
  });
  typeConverter.addConversion([](ColumnHandleType t) {
    return mlir::LLVM::LLVMPointerType::get(t.getContext());
  });
  typeConverter.addConversion([](PrintChunkType t) {
    return mlir::LLVM::LLVMPointerType::get(t.getContext());
  });
  typeConverter.addConversion([](PrintHandleType t) {
    return mlir::LLVM::LLVMPointerType::get(t.getContext());
  });
  typeConverter.addConversion([&](StructType t) -> mlir::Type {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(typeConverter.convertTypes(t.getFieldTypes(), types))) {
      return nullptr;
    }

    return mlir::LLVM::LLVMStructType::getLiteral(t.getContext(), types);
  });
  typeConverter.addConversion([](StringLiteralType t) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(t.getContext());
  });

  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateSCFToControlFlowConversionPatterns(patterns);

  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  patterns.add<AllocStructOpLowering, GetStructElementOpLowering,
               ConstantStringOpLowering>(typeConverter);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar