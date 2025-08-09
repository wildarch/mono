#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
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

class TypeConverter : public mlir::LLVMTypeConverter {
private:
  template <typename T> void addConversionToPointer() {
    addConversion([](T type) {
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    });
  }

  template <typename T1, typename T2, typename... Ts>
  void addConversionToPointer() {
    addConversionToPointer<T1>();
    addConversionToPointer<T2, Ts...>();
  }

public:
  TypeConverter(mlir::MLIRContext *ctx);
};

template <typename SourceOp>
class OpLowering : public mlir::ConvertOpToLLVMPattern<SourceOp> {
  using mlir::ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

} // namespace

TypeConverter::TypeConverter(mlir::MLIRContext *ctx)
    : mlir::LLVMTypeConverter(ctx) {
  addConversionToPointer<PointerType, ScannerHandleType, ColumnHandleType,
                         PrintChunkType, PrintHandleType, PipelineContextType,
                         StringLiteralType, ByteArrayType, TupleBufferLocalType,
                         TupleBufferType, AllocatorType>();
  addConversion([&](StructType t) -> mlir::Type {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(convertTypes(t.getFieldTypes(), types))) {
      return nullptr;
    }

    return mlir::LLVM::LLVMStructType::getLiteral(t.getContext(), types);
  });
}

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

template <>
mlir::LogicalResult OpLowering<AllocStructOp>::matchAndRewrite(
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

template <>
mlir::LogicalResult OpLowering<GetStructElementOp>::matchAndRewrite(
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

template <>
mlir::LogicalResult OpLowering<ConstantStringOp>::matchAndRewrite(
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
    op->emitOpError("cannot convert input types");
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
      op->emitOpError("cannot convert result types");
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
    if (!type) {
      hadError = true;
    }

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

static mlir::SymbolRefAttr makeInitFunction(GlobalOp op, std::size_t idx,
                                            mlir::IRRewriter &rewriter) {
  auto name = llvm::formatv("global{}_init", idx).sstr<32>();
  auto &block = op.getInit().front();

  // Change terminator to ReturnOp appropriate for FuncOp.
  mlir::func::ReturnOp returnOp;
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto globalReturn = llvm::cast<GlobalReturnOp>(block.getTerminator());
    rewriter.setInsertionPoint(globalReturn);
    // Write to global.
    rewriter.create<GlobalWriteOp>(globalReturn.getLoc(), op.getSymName(),
                                   globalReturn.getInput());
    returnOp = rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
        globalReturn, mlir::ValueRange{});
  }

  auto funcType =
      rewriter.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
  auto funcOp =
      rewriter.create<mlir::func::FuncOp>(op.getLoc(), name, funcType);
  rewriter.inlineRegionBefore(op.getInit(), funcOp.getBody(),
                              funcOp.getBody().begin());

  return mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr());
}

static mlir::SymbolRefAttr makeDestroyFunction(GlobalOp op, std::size_t idx,
                                               mlir::IRRewriter &rewriter) {
  auto name = llvm::formatv("global{}_destroy", idx).sstr<32>();
  auto &block = op.getDestroy().front();

  // Change terminator to ReturnOp appropriate for FuncOp.
  mlir::func::ReturnOp returnOp;
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto globalReturn = llvm::cast<GlobalReturnOp>(block.getTerminator());
    rewriter.setInsertionPoint(globalReturn);
    returnOp = rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
        globalReturn, mlir::ValueRange{});
  }

  auto funcType =
      rewriter.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
  auto funcOp =
      rewriter.create<mlir::func::FuncOp>(op.getLoc(), name, funcType);
  rewriter.inlineRegionBefore(op.getInit(), funcOp.getBody(),
                              funcOp.getBody().begin());

  // Read the state variable.
  mlir::Value state;
  auto &newBlock = funcOp.getBody().emplaceBlock();
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&newBlock);
    state = rewriter.create<GlobalReadOp>(op.getLoc(), op.getGlobalType(),
                                          op.getSymName());
  }

  // Inline the destroy block, passing in the state.
  rewriter.inlineBlockBefore(&block, &newBlock, newBlock.end(), state);
  return mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr());
}

static void globalMakeRef(GlobalOp op, std::size_t idx,
                          mlir::IRRewriter &rewriter) {
  auto initSym = makeInitFunction(op, idx, rewriter);
  auto destroySym = makeDestroyFunction(op, idx, rewriter);

  rewriter.create<GlobalRefOp>(op.getLoc(), initSym, destroySym);
}

static void pipelineMakeRef(PipelineLowOp op, std::size_t idx,
                            mlir::IRRewriter &rewriter) {
  auto globalOpenSym =
      moveToFunction(op.getGlobalOpen(), idx, "globalOpen", rewriter);
  auto localOpenSym =
      moveToFunction(op.getLocalOpen(), idx, "localOpen", rewriter);
  auto bodySym = moveToFunction(op.getBody(), idx, "body", rewriter);
  auto localCloseSym =
      moveToFunction(op.getLocalClose(), idx, "localClose", rewriter);
  auto globalCloseSym =
      moveToFunction(op.getGlobalClose(), idx, "globalClose", rewriter);

  rewriter.replaceOpWithNewOp<PipelineRefOp>(
      op, globalOpenSym, localOpenSym, bodySym, localCloseSym, globalCloseSym);
}

template <>
mlir::LogicalResult OpLowering<GlobalReadOp>::matchAndRewrite(
    GlobalReadOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(op.getGlobalName());
  if (!global) {
    return op.emitOpError("global variable not found: ") << op.getGlobalName();
  }

  auto globalPtr =
      rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), global);
  auto resultType = typeConverter->convertType(op.getResult().getType());
  if (!resultType) {
    return op.emitOpError("cannot convert result type");
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, resultType, globalPtr);
  return mlir::success();
}

template <>
mlir::LogicalResult OpLowering<GlobalWriteOp>::matchAndRewrite(
    GlobalWriteOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(op.getGlobalName());
  if (!global) {
    return op.emitOpError("global variable not found: ") << op.getGlobalName();
  }

  auto globalPtr =
      rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), global);
  auto resultType = typeConverter->convertType(op.getValue().getType());
  if (!resultType) {
    return op.emitOpError("cannot convert result type");
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(),
                                                   globalPtr);
  return mlir::success();
}

template <>
mlir::LogicalResult OpLowering<RuntimeCallOp>::matchAndRewrite(
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

template <>
mlir::LogicalResult OpLowering<GlobalOp>::matchAndRewrite(
    GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = typeConverter->convertType(op.getGlobalType());
  if (!type) {
    return op.emitOpError("unsupported global type: ") << op.getGlobalType();
  }

  mlir::Attribute value;
  if (llvm::isa<mlir::LLVM::LLVMPointerType>(type)) {
    value = rewriter.getZeroAttr(type);
  } else {
    return op.emitOpError("no default value for global of type: ") << type;
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
      op, type,
      /*isConstant=*/false, mlir::LLVM::Linkage::Internal, op.getName(), value);

  return mlir::success();
}

template <>
mlir::LogicalResult OpLowering<GetFieldPtrOp>::matchAndRewrite(
    GetFieldPtrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultType = rewriter.getType<mlir::LLVM::LLVMPointerType>();
  auto structType = llvm::cast<StructType>(op.getBase().getType().getPointee());
  auto llvmStructType = typeConverter->convertType(structType);
  if (!llvmStructType) {
    return op.emitOpError("cannot convert struct type: ") << structType;
  }

  mlir::LLVM::GEPArg indices[] = {
      // First struct (not an array, just a pointer to a single struct).
      mlir::LLVM::GEPArg(0),
      // This field of the struct
      mlir::LLVM::GEPArg(op.getField())};
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, resultType, llvmStructType,
                                                 adaptor.getBase(), indices);
  return mlir::success();
}

template <>
mlir::LogicalResult OpLowering<TypeSizeOp>::matchAndRewrite(
    TypeSizeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = typeConverter->convertType(op.getType());
  if (!type) {
    return op.emitOpError("cannot convert type: ") << type;
  }

  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto dataLayout = mlir::DataLayout(module);
  auto typeSize = dataLayout.getTypeSize(type);
  if (!typeSize.isFixed()) {
    return op.emitOpError("type has variable size: ") << type;
  }

  auto constantOp = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(typeSize.getFixedValue()));

  rewriter.replaceOp(op, constantOp);
  return mlir::success();
}

template <>
mlir::LogicalResult OpLowering<TypeAlignOp>::matchAndRewrite(
    TypeAlignOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = typeConverter->convertType(op.getType());
  if (!type) {
    return op.emitOpError("cannot convert type: ") << type;
  }

  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto dataLayout = mlir::DataLayout(module);
  auto typeAlign = dataLayout.getTypeABIAlignment(type);

  auto constantOp = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(),
      rewriter.getI64IntegerAttr(typeAlign));

  rewriter.replaceOp(op, constantOp);
  return mlir::success();
}

void LowerToLLVM::runOnOperation() {
  TypeConverter typeConverter(&getContext());

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

  // Turn global initializers and destructors into functions, and create
  // GlobalRefOps to track them.
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<GlobalOp> globals(getOperation().getOps<GlobalOp>());
  for (auto [i, op] : llvm::enumerate(globals)) {
    rewriter.setInsertionPointAfter(op);
    globalMakeRef(op, i, rewriter);
  }

  // Replace pipeline blocks with references to functions.
  llvm::SmallVector<PipelineLowOp> pipelines(
      getOperation().getOps<PipelineLowOp>());
  for (auto [i, op] : llvm::enumerate(pipelines)) {
    rewriter.setInsertionPoint(op);
    pipelineMakeRef(op, i, rewriter);
  }

  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<PipelineRefOp>();
  target.addLegalOp<GlobalRefOp>();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateSCFToControlFlowConversionPatterns(patterns);

  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  patterns.add<OpLowering<AllocStructOp>, OpLowering<GetStructElementOp>,
               OpLowering<ConstantStringOp>, OpLowering<RuntimeCallOp>,
               OpLowering<GlobalOp>, OpLowering<GlobalReadOp>,
               OpLowering<GlobalWriteOp>, OpLowering<GetFieldPtrOp>,
               OpLowering<TypeSizeOp>, OpLowering<TypeAlignOp>>(typeConverter);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar
