#include <iostream>

#include "MiniDialect.h"
#include "MiniLoweringPass.h"
#include "MiniOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "mini tool\n");

  mlir::MLIRContext context;
  context.getOrLoadDialect<experiments_mlir::mini::MiniDialect>();

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Add a main function
  llvm::SmallVector<mlir::Type> resTypes = {builder.getI32Type()};
  auto funcType = builder.getFunctionType(std::nullopt, resTypes);
  auto mainFuncOp = builder.create<experiments_mlir::mini::FuncOp>(
      builder.getUnknownLoc(), "main", funcType);

  mlir::Block &entryBlock = mainFuncOp.front();
  // TODO: arguments
  builder.setInsertionPointToStart(&entryBlock);

  auto constantOp = builder.create<experiments_mlir::mini::ConstantOp>(
      builder.getUnknownLoc(), 42);

  builder.create<experiments_mlir::mini::ReturnOp>(builder.getUnknownLoc(),
                                                   constantOp);

  module.dump();

  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);
  auto lowerPass = experiments_mlir::mini::createMiniLoweringPass();
  pm.addPass(std::move(lowerPass));

  if (mlir::failed(pm.run(module))) {
    std::cerr << "Failed to lower!" << std::endl;
    return 1;
  }

  module.dump();

  // Reuse the pass manager for the next step: lowering to LLVM IR.
  pm.clear();
  pm.addPass(experiments_mlir::mini::createLowerToLLVMPass());

  if (mlir::failed(pm.run(module))) {
    std::cerr << "Failed to lower to LLVM!" << std::endl;
    return 1;
  }

  module.dump();

  // JIT compile and run it
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*module.getContext());
  auto optPipeline = mlir::makeOptimizingTransformer(
      /* optLevel*/ 0, /* sizeLevel */ 0, /* targetMachine */ nullptr);
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto engine = maybeEngine->get();

  int32_t result = 0;
  auto resultWrapper = mlir::ExecutionEngine::result(result);
  llvm::SmallVector<void *, 1> args;
  mlir::ExecutionEngine::Argument<mlir::ExecutionEngine::Result<int32_t>>::pack(
      args, resultWrapper);
  if (engine->invokePacked("main", args)) {
    llvm::errs() << "JIT invocation failed\n";
    return 1;
  }

  llvm::outs() << "Result: " << result << "\n";

  return 0;
}