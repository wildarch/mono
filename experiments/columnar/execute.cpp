#include <columnar/Columnar.h>
#include <iostream>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

static int loadMLIR(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

static int runJit(mlir::ModuleOp module) {
  // Find pipelines to run
  llvm::SmallVector<columnar::PipelineRefOp> pipelines;
  module->walk([&](columnar::PipelineRefOp op) { pipelines.push_back(op); });

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (auto err = maybeEngine.takeError()) {
    llvm::errs() << "Failed to construct an execution engine: " << err << "\n";
    return -1;
  }

  auto &engine = maybeEngine.get();

  // Run pipelines to completion
  for (auto pipe : pipelines) {
    // Initialize global state.
    void *globalState;
    if (auto func = pipe.getGlobalOpen()) {
      if (auto err = engine->invoke(func->getLeafReference(),
                                    engine->result(globalState))) {
        llvm::errs() << "Failed to run globalOpen for pipeline: " << err
                     << "\n";
        return -1;
      }
    }

    // Run body repeatedly.
    bool haveMore = true;
    while (haveMore) {
      if (auto err = engine->invoke(pipe.getBody().getLeafReference(),
                                    engine->result(haveMore))) {
        llvm::errs() << "Failed to run body for pipeline: " << err << "\n";
        return -1;
      }
    }

    // Free global state.
    if (auto func = pipe.getGlobalClose()) {
      if (auto err = engine->invoke(func->getLeafReference(), globalState)) {
        llvm::errs() << "Failed to run globalClose for pipeline: " << err
                     << "\n";
        return -1;
      }
    }
  }

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  mlir::DialectRegistry registry;
  columnar::registerLLVMTranslation(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<columnar::ColumnarDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int err = loadMLIR(context, module)) {
    return err;
  }

  if (int err = runJit(*module)) {
    return err;
  }

  return 0;
}