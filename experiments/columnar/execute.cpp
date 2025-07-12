#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "columnar/runtime/PipelineContext.h"
#include <columnar/Columnar.h>
#include <columnar/Runtime.h>

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

  // Bind runtime functions
  engine->registerSymbols(columnar::registerRuntimeSymbols);

  using GlobalOpenFunc = void *();
  using LocalOpenFunc = void *(void *);
  using BodyFunc = bool(columnar::runtime::PipelineContext *, void *, void *);
  using LocalCloseFunc = void *(void *, void *);
  using GlobalCloseFunc = void(void *);

  // Run pipelines to completion
  for (auto pipe : pipelines) {
    // Initialize global state.
    void *globalState;
    if (auto func = pipe.getGlobalOpen()) {
      auto maybeFuncPtr = engine->lookup(func->getLeafReference());
      if (auto err = maybeFuncPtr.takeError()) {
        llvm::errs() << "Failed to lookup globalOpen for pipeline: " << err
                     << "\n";
        return -1;
      }

      auto *funcPtr = ((GlobalOpenFunc *)maybeFuncPtr.get());
      globalState = funcPtr();
    }

    // Initialize local state.
    void *localState;
    if (auto func = pipe.getLocalOpen()) {
      auto maybeFuncPtr = engine->lookup(func->getLeafReference());
      if (auto err = maybeFuncPtr.takeError()) {
        llvm::errs() << "Failed to lookup localOpen for pipeline: " << err
                     << "\n";
        return -1;
      }

      auto *funcPtr = ((LocalOpenFunc *)maybeFuncPtr.get());
      localState = funcPtr(globalState);
    }

    // Run body repeatedly.
    auto maybeBodyFunc = engine->lookup(pipe.getBody().getLeafReference());
    if (auto err = maybeBodyFunc.takeError()) {
      llvm::errs() << "Cannot find body function: " << err << "\n";
      return -1;
    }

    auto *bodyFunc = ((BodyFunc *)maybeBodyFunc.get());
    columnar::runtime::PipelineContext pipelineCtx;
    bool haveMore = true;
    while (haveMore) {
      haveMore = bodyFunc(&pipelineCtx, globalState, localState);
    }

    // Free local state.
    if (auto func = pipe.getLocalClose()) {
      auto maybeFuncPtr = engine->lookup(func->getLeafReference());
      if (auto err = maybeFuncPtr.takeError()) {
        llvm::errs() << "Failed to lookup localClose for pipeline: " << err
                     << "\n";
        return -1;
      }

      auto *funcPtr = ((LocalCloseFunc *)maybeFuncPtr.get());
      funcPtr(globalState, localState);
    }

    // Free global state.
    if (auto func = pipe.getGlobalClose()) {
      auto maybeFuncPtr = engine->lookup(func->getLeafReference());
      if (auto err = maybeFuncPtr.takeError()) {
        llvm::errs() << "Failed to lookup globalClose for pipeline: " << err
                     << "\n";
        return -1;
      }

      auto *funcPtr = ((GlobalCloseFunc *)maybeFuncPtr.get());
      funcPtr(globalState);
    }
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
