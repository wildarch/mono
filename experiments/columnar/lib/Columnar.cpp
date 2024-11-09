#include "columnar/Columnar.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "columnar/Dialect.cpp.inc"

#include "columnar/Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "columnar/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "columnar/Types.cpp.inc"

#define GET_OP_CLASSES
#include "columnar/Ops.cpp.inc"

namespace columnar {

mlir::Type getCountElementType(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, /*width=*/64, mlir::IntegerType::Signed);
}

void ColumnarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "columnar/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "columnar/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "columnar/Attrs.cpp.inc"
      >();
}

mlir::LogicalResult QueryOutputOp::verify() {
  if (getColumns().size() != getNames().size()) {
    return emitOpError("Outputs ")
           << getColumns().size() << " columns, but got " << getNames().size()
           << " column names";
  }

  return mlir::success();
}

static ColumnType aggregatorType(ColumnType input, Aggregator agg) {
  switch (agg) {
  // Same return type regardless of input.
  case Aggregator::COUNT:
  case Aggregator::COUNT_DISTINCT: {
    auto *ctx = input.getContext();
    return ColumnType::get(ctx, getCountElementType(ctx));
  }
  // Monoids, same output type as input
  case Aggregator::SUM:
  case Aggregator::AVG:
  case Aggregator::MIN:
    return input;
  }

  llvm_unreachable("invalid enum");
}

mlir::LogicalResult AggregateOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // Keep types from the group-by
  for (auto groupBy : adaptor.getGroupBy()) {
    inferredReturnTypes.push_back(groupBy.getType());
  }

  for (auto [col, agg] :
       llvm::zip_equal(adaptor.getAggregate(), adaptor.getAggregators())) {
    inferredReturnTypes.push_back(
        aggregatorType(llvm::cast<ColumnType>(col.getType()), agg));
  }

  return mlir::success();
}

mlir::LogicalResult AggregateOp::verify() {
  if (getAggregate().size() != getAggregators().size()) {
    return emitOpError("has ")
           << getAggregate().size() << " columns to aggregate, but "
           << getAggregators().size() << " aggregators";
  }

  return mlir::success();
}

mlir::LogicalResult JoinOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  for (auto col : adaptor.getLhs()) {
    inferredReturnTypes.push_back(col.getType());
  }

  for (auto col : adaptor.getRhs()) {
    inferredReturnTypes.push_back(col.getType());
  }

  return mlir::success();
}

} // namespace columnar