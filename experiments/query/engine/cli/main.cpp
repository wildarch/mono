#include <arrow/api.h>
#include <iostream>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Parser/Parser.h>
#include <parquet/file_reader.h>
#include <stdexcept>

#include "execution/Batch.h"
#include "execution/Common.h"
#include "execution/ImplementationGenerator.h"
#include "execution/ParquetScanner.h"
#include "execution/expression/BinaryOperatorExpression.h"
#include "execution/expression/ColumnExpression.h"
#include "execution/expression/ConstantExpression.h"
#include "execution/expression/IR/ExpressionDialect.h"
#include "execution/operator/IR/OperatorDialect.h"
#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/IR/OperatorTypes.h"
#include "execution/operator/impl/FilterOperator.h"
#include "execution/operator/impl/Operator.h"
#include "execution/operator/impl/ParquetScanOperator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

int main(int argc, char **argv) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<execution::qoperator::OperatorDialect>();
  ctx.getOrLoadDialect<execution::expression::ExpressionDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  if (argc != 2) {
    std::cerr << "Expect path to mlir query file" << std::endl;
    return 1;
  }

  auto sourceFile = llvm::MemoryBuffer::getFile(argv[1]);
  if (auto ec = sourceFile.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*sourceFile), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);

  llvm::outs() << "Module:\n";
  module->print(llvm::outs());

  // TODO: tidy this up
  auto root = module->getBody()->getTerminator();
  if (!root) {
    llvm::errs() << "No root operator\n";
    return -1;
  }
  llvm::outs() << "root op: ";
  root->print(llvm::outs());
  llvm::outs() << "\n";

  // TEMP
  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointAfter(root);
  auto agg = builder.create<execution::qoperator::AggregateOp>(
      builder.getUnknownLoc(), root->getResult(0).getType(),
      root->getResult(0));
  auto &aggBlock = agg.getAggregators().emplaceBlock();
  auto arg =
      aggBlock.addArgument(builder.getI32Type(), builder.getUnknownLoc());
  builder.setInsertionPointToStart(&aggBlock);
  auto key = builder.create<execution::qoperator::AggregateKeyOp>(
      builder.getUnknownLoc(),
      builder.getType<execution::qoperator::AggregatorType>(arg.getType()),
      arg);
  builder.create<execution::qoperator::AggregateReturnOp>(
      builder.getUnknownLoc(), mlir::ValueRange{key, key});
  agg->print(llvm::outs());

  auto rootImpl = execution::generateImplementation(root);
  if (!rootImpl) {
    llvm::errs() << "Failed to implement query\n";
    return -1;
  }

  /*
  int64_t sum = 0;
  std::optional<execution::Batch> batch;
  while ((batch = rootImpl->poll())) {
    auto &column = batch->columns().at(0);
    for (auto val : column.get<execution::PhysicalColumnType::INT32>()) {
      sum += val;
    }
  }
  std::cout << "sum: " << sum << "\n";
  */

  std::optional<execution::Batch> batch;
  while ((batch = rootImpl->poll())) {
    auto &column = batch->columns().at(0);
    for (auto val : column.get<execution::PhysicalColumnType::STRING>()) {
      std::cout << "string: " << std::string_view(val) << "\n";
    }
  }

  return 0;
}