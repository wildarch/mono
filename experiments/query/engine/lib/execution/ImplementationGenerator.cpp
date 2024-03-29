#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ErrorOr.h>
#include <mlir/IR/Operation.h>
#include <parquet/file_reader.h>
#include <parquet/schema.h>

#include "execution/Batch.h"
#include "execution/ImplementationGenerator.h"
#include "execution/ParquetScanner.h"
#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/IR/OperatorTypes.h"
#include "execution/operator/impl/AggregateOperator.h"
#include "execution/operator/impl/FilterOperator.h"
#include "execution/operator/impl/Operator.h"
#include "execution/operator/impl/ParquetScanOperator.h"
#include "execution/operator/impl/ProjectOperator.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace {

class ImplementationGenerator {
private:
  llvm::DenseMap<mlir::Operation *, execution::OperatorPtr> _implementations;

public:
  execution::OperatorPtr implement(mlir::Operation *op);
  execution::OperatorPtr implement(execution::qoperator::ScanParquetOp op);

  execution::PhysicalColumnType convert(mlir::Type type);
};

} // namespace

execution::OperatorPtr ImplementationGenerator::implement(mlir::Operation *op) {
  using namespace execution::qoperator;
  return llvm::TypeSwitch<mlir::Operation *, execution::OperatorPtr>(op)
      .Case<AggregateOp>([this](AggregateOp op) {
        auto child = implement(op.getChild().getDefiningOp());
        auto expr = llvm::cast<AggregateReturnOp>(
            op.getAggregators().front().getTerminator());
        return std::make_shared<execution::AggregateOperator>(child, expr);
      })
      .Case<FilterOp>([this](FilterOp op) {
        auto child = implement(op.getChild().getDefiningOp());
        auto expr = llvm::cast<FilterReturnOp>(
            op.getPredicate().front().getTerminator());
        return std::make_shared<execution::FilterOperator>(child, expr);
      })
      .Case<ScanParquetOp>([this](ScanParquetOp op) { return implement(op); })
      .Case<ProjectOp>([this](ProjectOp op) {
        auto child = implement(op.getChild().getDefiningOp());
        auto expr = llvm::cast<ProjectReturnOp>(
            op.getProjections().front().getTerminator());
        return std::make_shared<execution::ProjectOperator>(child, expr);
      })
      .Default([](mlir::Operation *op) {
        op->emitOpError("cannot implement");
        return nullptr;
      });
}

execution::OperatorPtr
ImplementationGenerator::implement(execution::qoperator::ScanParquetOp op) {
  auto reader = parquet::ParquetFileReader::OpenFile(op.getPath().str());

  auto schema = reader->metadata()->schema();

  llvm::SmallVector<execution::ParquetScanner::ColumnToRead> columns;
  for (const auto &[nameAttr, type] :
       llvm::zip_equal(op.getColumns(), op.getType().getColumns())) {
    auto name = llvm::cast<mlir::StringAttr>(nameAttr);
    auto index = schema->ColumnIndex(name.str());
    if (index < 0) {
      throw std::invalid_argument("column does not exist");
    }
    columns.emplace_back(execution::ParquetScanner::ColumnToRead{
        .columnId = index, .type = convert(type)});
  }

  return std::make_shared<execution::ParquetScanOperator>(std::move(reader),
                                                          columns);
}

execution::PhysicalColumnType
ImplementationGenerator::convert(mlir::Type type) {
  if (type.isInteger(/*width=*/32)) {
    return execution::PhysicalColumnType::INT32;
  }
  if (type.isInteger(/*width=*/64)) {
    return execution::PhysicalColumnType::INT64;
  }

  if (type.isa<execution::qoperator::VarcharType>()) {
    return execution::PhysicalColumnType::STRING;
  }

  type.dump();
  llvm_unreachable("cannot convert to physical column type");
}

namespace execution {

OperatorPtr generateImplementation(mlir::Operation *root) {
  ImplementationGenerator gen;
  return gen.implement(root);
}

} // namespace execution