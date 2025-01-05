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

namespace {

class ColumnarOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute attr,
                       mlir::raw_ostream &os) const override {
    // Add aliases in the form catalog_<oid>_<type> as we cannot end the
    // alias with a digit
    if (auto info = llvm::dyn_cast<TableAttr>(attr)) {
      os << "table_" << info.getName();
      return AliasResult::FinalAlias;
    } else if (auto info = llvm::dyn_cast<TableColumnAttr>(attr)) {
      os << "column_" << info.getTable().getName() << "_" << info.getName();
      return AliasResult::FinalAlias;
    }

    return AliasResult::NoAlias;
  }
};

} // namespace

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

  addInterfaces<ColumnarOpAsmDialectInterface>();
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
  } else if (auto table = llvm::dyn_cast<TableAttr>(value)) {
    auto colType = llvm::dyn_cast<ColumnType>(rawType);
    if (colType && llvm::isa<SelectType>(colType.getElementType())) {
      return builder.create<SelTableOp>(loc, table);
    }
  }

  mlir::emitError(loc) << "cannot materialize constant " << value
                       << " with type " << type;
  return nullptr;
}

mlir::Type DecimalAttr::getType() { return DecimalType::get(getContext()); }

mlir::Type StringAttr::getType() { return StringType::get(getContext()); }

mlir::Type DateAttr::getType() { return DateType::get(getContext()); }

mlir::Type SelIdAttr::getType() { return SelectType::get(getContext()); }

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
  case Aggregator::COUNT_ALL:
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

mlir::ValueRange JoinOp::getLhsResults() {
  return getResults().take_front(getLhs().size());
}

mlir::ValueRange JoinOp::getRhsResults() {
  return getResults().drop_front(getLhs().size());
}

void SelectOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::ValueRange inputs) {
  build(builder, state, inputs.getTypes(), inputs);
  auto &block = state.regions[0]->emplaceBlock();

  // Populate block arguments to match inputs.
  for (auto input : inputs) {
    block.addArgument(input.getType(), input.getLoc());
  }
}

mlir::LogicalResult
SelectOp::fold(FoldAdaptor adaptor,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
  if (getPredicates().front().empty()) {
    // No filters applied, remove.
    for (auto input : getInputs()) {
      results.push_back(input);
    }

    return mlir::success();
  }

  return mlir::failure();
}

mlir::LogicalResult
PredicateOp::fold(FoldAdaptor adaptor,
                  llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
  // Remove inputs that we don't use.
  llvm::BitVector erase(getNumOperands());
  for (auto arg : getBody().getArguments()) {
    if (arg.use_empty()) {
      erase.set(arg.getArgNumber());
    }
  }

  if (erase.any()) {
    llvm::SmallVector<mlir::Value> inputs;
    for (auto [i, v] : llvm::enumerate(getInputs())) {
      if (!erase[i]) {
        inputs.push_back(v);
      }
    }

    getInputsMutable().assign(inputs);
    getBody().front().eraseArguments(erase);
    return mlir::success();
  }

  return mlir::failure();
}

mlir::LogicalResult OrderByOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  for (auto col : adaptor.getKeys()) {
    inferredReturnTypes.push_back(col.getType());
  }

  for (auto col : adaptor.getValues()) {
    inferredReturnTypes.push_back(col.getType());
  }

  return mlir::success();
}

mlir::ValueRange OrderByOp::getKeyResults() {
  return getResults().take_front(getKeys().size());
}

mlir::ValueRange OrderByOp::getValueResults() {
  return getResults().drop_front(getKeys().size());
}

mlir::LogicalResult LimitOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  auto types = adaptor.getInputs().getTypes();
  inferredReturnTypes.append(types.begin(), types.end());
  return mlir::success();
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

mlir::LogicalResult SelAddOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  auto types = adaptor.getInputs().getTypes();
  inferredReturnTypes.append(types.begin(), types.end());

  inferredReturnTypes.push_back(ColumnType::get(SelectType::get(ctx)));
  return mlir::success();
}

mlir::OpFoldResult SelTableOp::fold(FoldAdaptor adaptor) { return getTable(); }

} // namespace columnar