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

  auto rootImpl = execution::generateImplementation(root);
  if (!rootImpl) {
    llvm::errs() << "Failed to implement query\n";
    return -1;
  }

  std::optional<execution::Batch> batch;
  llvm::outs() << "l_returnflag"
               << ",l_linestatus"
               << ",sum_qty"
               << ",sum_base_price"
               << ",sum_disc_price"
               << ",sum_charge"
               << ",count_order"
               << "\n";
  while ((batch = rootImpl->poll())) {
    for (uint32_t row = 0; row < batch->rows(); row++) {
      auto returnFlag =
          batch->columns()[0].get<execution::PhysicalColumnType::STRING>()[row];
      auto lineStatus =
          batch->columns()[1].get<execution::PhysicalColumnType::STRING>()[row];
      auto sumQty =
          batch->columns()[2].get<execution::PhysicalColumnType::INT64>()[row];
      auto sumBasePrice =
          batch->columns()[3].get<execution::PhysicalColumnType::INT64>()[row];
      auto sumDiscPrice =
          batch->columns()[4].get<execution::PhysicalColumnType::INT64>()[row];
      auto sumCharge =
          batch->columns()[5].get<execution::PhysicalColumnType::INT64>()[row];
      auto countOrder = batch->columns()
                            .back()
                            .get<execution::PhysicalColumnType::INT64>()[row];
      llvm::outs() << returnFlag << "," << lineStatus << "," << sumQty << ","
                   << sumBasePrice << "," << sumDiscPrice << "," << sumCharge
                   << "," << countOrder << "\n";
    }
  }

  return 0;
}