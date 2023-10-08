#include "execution/operator/impl/AggregateOperator.h"
#include "execution/Batch.h"
#include "execution/expression/ExpressionEvaluator.h"
#include "execution/operator/IR/OperatorOps.h"
#include <iostream>
#include <llvm-17/llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/Hashing.h>

namespace execution {

size_t
AggregateOperator::Hasher::operator()(const std::vector<AnyValue> &keys) const {
  // TODO: consider skipping collection to a vector.
  llvm::SmallVector<size_t> keyHashes;
  keyHashes.reserve(keys.size());
  for (const auto &key : keys) {
    keyHashes.emplace_back(std::hash<AnyValue>{}(key));
  }
  return llvm::hash_combine_range(keyHashes.begin(), keyHashes.end());
}

AggregateOperator::AggregateOperator(OperatorPtr child,
                                     qoperator::AggregateReturnOp retOp)
    : SingleChildOperator(OperatorKind::AGGREGATE, child), _retOp(retOp) {
  for (const auto &val : _retOp.getAggregators()) {
    auto aggType = llvm::cast<qoperator::AggregatorType>(val.getType());
    auto physType = mlirToPhysicalType(aggType.getElementType());
    _outputColumnTypes.emplace_back(physType);
    llvm::TypeSwitch<mlir::Operation *, void>(val.getDefiningOp())
        .Case<qoperator::AggregateKeyOp>([&](qoperator::AggregateKeyOp op) {
          _aggregateModes.emplace_back(AggregateOperator::AggregateMode::KEY);
        })
        .Case<qoperator::AggregateSumOp>([&](qoperator::AggregateSumOp op) {
          _aggregateModes.emplace_back(AggregateOperator::AggregateMode::SUM);
        })
        .Case<qoperator::AggregateCountOp>([&](qoperator::AggregateCountOp op) {
          // TODO: special case for counting
          _aggregateModes.emplace_back(AggregateOperator::AggregateMode::SUM);
        })
        .Default([](mlir::Operation *op) {
          op->dump();
          llvm_unreachable("not implemented");
        });
  }
}

void AggregateOperator::handleInputRow(const Batch &input, size_t row) {
  std::vector<AnyValue> keys;
  std::vector<AnyValue> values;
  for (const auto &val : _retOp.getAggregators()) {
    llvm::TypeSwitch<mlir::Operation *, void>(val.getDefiningOp())
        .Case<qoperator::AggregateKeyOp>([&](qoperator::AggregateKeyOp op) {
          keys.emplace_back(evaluate(op.getColumn(), input, row));
        })
        .Case<qoperator::AggregateSumOp>([&](qoperator::AggregateSumOp op) {
          values.emplace_back(evaluate(op.getColumn(), input, row));
        })
        .Case<qoperator::AggregateCountOp>([&](qoperator::AggregateCountOp op) {
          // TODO: special case for counting
          values.emplace_back(std::int64_t(1));
        })
        .Default([](mlir::Operation *op) {
          op->dump();
          llvm_unreachable("not implemented");
        });
  }

  auto &curValues = _aggregate[keys];
  if (curValues.empty()) {
    // New row
    curValues = values;
    return;
  }

  // Aggregate
  size_t valIdx = 0;
  for (size_t i = 0; i < _aggregateModes.size(); i++) {
    switch (_aggregateModes[i]) {
    case AggregateMode::KEY:
      // skip
      break;
    case AggregateMode::SUM:
      std::visit(
          AnyValueVisitor{
              [](const SmallString &s) {
                llvm_unreachable("cannot SUM string");
              },
              [&](const auto &v) {
                curValues[valIdx] =
                    std::get<std::decay_t<decltype(v)>>(curValues[valIdx]) + v;
              },
          },
          values[valIdx]);
      valIdx++;
      break;
    }
  }
}

void AggregateOperator::handleInputBatch(const Batch &input) {
  for (size_t row = 0; row < input.rows(); row++) {
    handleInputRow(input, row);
  }
}

void AggregateOperator::drainChild() {
  while (true) {
    auto input = child()->poll();
    if (!input) {
      break;
    }
    handleInputBatch(*input);
  }
}

std::optional<Batch> AggregateOperator::poll() {
  if (!_aggregateIter) {
    drainChild();
    _aggregateIter = _aggregate.begin();
  }

  if (*_aggregateIter == _aggregate.end()) {
    return std::nullopt;
  }

  static constexpr uint32_t MAX_ROWS_PER_BATCH = 1024;
  uint32_t rowsLeft = _aggregate.size() - _rowsSubmitted;
  uint32_t batchSize = std::min(MAX_ROWS_PER_BATCH, rowsLeft);
  Batch batch(_outputColumnTypes, batchSize);

  for (size_t row = 0; row < batchSize; row++) {
    const auto &[keys, values] = **_aggregateIter;
    size_t keyIdx = 0;
    size_t valIdx = 0;
    for (size_t col = 0; col < _outputColumnTypes.size(); col++) {
      // Get the value for this column
      AnyValue value;
      switch (_aggregateModes[col]) {
      case AggregateMode::KEY:
        value = keys[keyIdx];
        keyIdx++;
        break;
      case AggregateMode::SUM:
        value = values[valIdx];
        valIdx++;
        break;
      }

      switch (_outputColumnTypes[col]) {
#define CASE(t)                                                                \
  case PhysicalColumnType::t:                                                  \
    batch.columnsForWrite()[col].getForWrite<PhysicalColumnType::t>()[row] =   \
        std::get<StoredType<PhysicalColumnType::t>::type>(value);              \
    break;

        CASE(INT32)
        CASE(INT64)
        CASE(DOUBLE)
        CASE(STRING)
#undef CASE
      }
    }

    (*_aggregateIter)++;
  }

  _rowsSubmitted += batchSize;
  return batch;
}

} // namespace execution