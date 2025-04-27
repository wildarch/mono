#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include "columnar/Columnar.h"

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

class runtimeCallOpLowering
    : public mlir::ConvertOpToLLVMPattern<RuntimeCallOp> {
  using mlir::ConvertOpToLLVMPattern<RuntimeCallOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RuntimeCallOp op, OpAdaptor adaptor,
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
  auto cst0 = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
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
      op.getLoc(), mallocOp->getFunctionType(), mallocOp->getSymNameAttr(),
      size);

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

static mlir::FunctionType
runtimeCallFunctionType(RuntimeCallOp op,
                        const mlir::TypeConverter typeConverter) {
  llvm::SmallVector<mlir::Type> inputs;
  if (mlir::failed(
          typeConverter.convertTypes(op.getInputs().getTypes(), inputs))) {
    return nullptr;
  }

  llvm::SmallVector<mlir::Type> results;
  if (op.getNumResults() > 1) {
    // Pass as out pointers.
    auto ptrType = mlir::LLVM::LLVMPointerType::get(op.getContext());
    inputs.append(op.getNumResults(), ptrType);
  } else {
    if (mlir::failed(
            typeConverter.convertTypes(op.getResultTypes(), results))) {
      return nullptr;
    }
  }

  return mlir::FunctionType::get(op.getContext(), inputs, results);
}

static mlir::LogicalResult
findRuntimeCallsIn(mlir::Operation *op,
                   llvm::DenseMap<mlir::StringAttr, mlir::FunctionType> &called,
                   const mlir::TypeConverter &typeConverter) {
  bool hadError = false;
  op->walk([&](RuntimeCallOp op) {
    auto type = runtimeCallFunctionType(op, typeConverter);
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

mlir::LogicalResult runtimeCallOpLowering::matchAndRewrite(
    RuntimeCallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<mlir::Type> resultTypes;
  if (mlir::failed(
          typeConverter->convertTypes(op.getResultTypes(), resultTypes))) {
    return mlir::failure();
  }

  if (resultTypes.size() <= 1) {
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getFuncAttr(), resultTypes, adaptor.getInputs());
    return mlir::success();
  }

  // Allocate room on the stack for the results.
  auto cst1 = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
  auto ptrType = rewriter.getType<mlir::LLVM::LLVMPointerType>();
  llvm::SmallVector<mlir::Value> outAllocs;
  for (auto type : resultTypes) {
    outAllocs.emplace_back(rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), ptrType, type, cst1));
  }

  // Make the call
  llvm::SmallVector<mlir::Value> inputs(adaptor.getInputs());
  inputs.append(outAllocs);
  rewriter.create<mlir::func::CallOp>(op.getLoc(), op.getFuncAttr(),
                                      mlir::TypeRange{}, inputs);

  // Load the out values
  llvm::SmallVector<mlir::Value> outValues;
  for (auto [type, ptrVal] : llvm::zip_equal(resultTypes, outAllocs)) {
    outValues.emplace_back(
        rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), type, ptrVal));
  }

  rewriter.replaceOp(op, outValues);
  return mlir::success();
}

void LowerToLLVM::runOnOperation() {
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

  // Find called runtime functions
  llvm::DenseMap<mlir::StringAttr, mlir::FunctionType> called;
  if (mlir::failed(findRuntimeCallsIn(getOperation(), called, typeConverter))) {
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

  // Replace pipeline blocks with references to functions.
  mlir::IRRewriter rewriter(&getContext());
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

  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateSCFToControlFlowConversionPatterns(patterns);

  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  patterns.add<AllocStructOpLowering, GetStructElementOpLowering,
               ConstantStringOpLowering, runtimeCallOpLowering>(typeConverter);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar
