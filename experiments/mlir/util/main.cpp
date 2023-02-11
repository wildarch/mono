#include <iostream>

#include "MiniDialect.h"
#include "MiniOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"

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
  llvm::SmallVector<mlir::Type> argTypes(0);
  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
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
}