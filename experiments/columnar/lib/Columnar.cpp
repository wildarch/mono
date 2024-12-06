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

static DateAttr parseDate(mlir::OpBuilder &builder, llvm::StringRef s) {
  std::uint16_t year;
  if (s.consumeInteger(10, year)) {
    return nullptr;
  }

  if (!s.consume_front("-")) {
    return nullptr;
  }

  std::uint8_t month;
  if (s.consumeInteger(10, month)) {
    return nullptr;
  }

  if (!s.consume_front("-")) {
    return nullptr;
  }

  std::uint8_t day;
  if (s.consumeInteger(10, day)) {
    return nullptr;
  }

  if (!s.empty()) {
    return nullptr;
  }

  return builder.getAttr<DateAttr>(year, month, day);
}

mlir::Operation *ColumnarDialect::materializeConstant(mlir::OpBuilder &builder,
                                                      mlir::Attribute value,
                                                      mlir::Type rawType,
                                                      mlir::Location loc) {
  auto type = llvm::cast<ColumnType>(rawType);
  if (auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(value)) {
    if (typedAttr.getType() == type.getElementType()) {
      return builder.create<ConstantOp>(loc, typedAttr);
    }

    if (llvm::isa<DecimalType>(type.getElementType()) &&
        typedAttr.getType().isSignedInteger(64)) {
      // Coerce int to decimal.
      auto intVal = llvm::cast<mlir::IntegerAttr>(value);
      auto decVal = (intVal.getValue() * 100).trySExtValue();
      if (!decVal) {
        mlir::emitError(loc) << "cannot coerce value " << intVal
                             << " to decimal because it is too large";
        return nullptr;
      }

      return builder.create<ConstantOp>(loc,
                                        builder.getAttr<DecimalAttr>(*decVal));
    }

    if (llvm::isa<DateType>(type.getElementType()) &&
        llvm::isa<StringType>(typedAttr.getType())) {
      // Date as string.
      auto sval = llvm::cast<StringAttr>(typedAttr).getValue().getValue();
      auto date = parseDate(builder, sval);
      if (!date) {
        mlir::emitError(loc) << "invalid date string '" << sval << "'";
        return nullptr;
      }

      return builder.create<ConstantOp>(loc, date);
    }
  }

  mlir::emitError(loc) << "cannot materialize constant " << value
                       << " with type " << type;
  return nullptr;
}

mlir::Type DecimalAttr::getType() { return DecimalType::get(getContext()); }

mlir::Type StringAttr::getType() { return StringType::get(getContext()); }

mlir::Type DateAttr::getType() { return DateType::get(getContext()); }

mlir::LogicalResult QueryOutputOp::verify() {
  if (getColumns().size() != getNames().size()) {
    return emitOpError("outputs ")
           << getColumns().size() << " columns, but got " << getNames().size()
           << " column names";
  }

  return mlir::success();
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

mlir::LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      ColumnType::get(ctx, adaptor.getValue().getType()));
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

mlir::ValueRange AggregateOp::getGroupByResults() {
  return getResults().take_front(getGroupBy().size());
}

mlir::ValueRange AggregateOp::getAggregationResults() {
  return getResults().drop_front(getGroupBy().size());
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

void SelectOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::ValueRange inputs) {
  build(builder, state, inputs.getTypes(), inputs);
}

mlir::Block &SelectOp::addPredicate() {
  auto &block = getPredicates().emplaceBlock();

  // Populate block arguments to match inputs.
  for (auto input : getInputs()) {
    block.addArgument(input.getType(), input.getLoc());
  }

  return block;
}

mlir::LogicalResult
SelectOp::fold(FoldAdaptor adaptor,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
  if (getPredicates().empty()) {
    // No filters applied, remove.
    for (auto input : getInputs()) {
      results.push_back(input);
    }
    return mlir::success();
  }

  return mlir::failure();
}

mlir::OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getInput().getType()) {
    return getInput();
  }

  if (auto attr = adaptor.getInput()) {
    return attr;
  }

  return nullptr;
}

} // namespace columnar