#pragma once

#include "execution/Batch.h"
#include "execution/expression/ExpressionEvaluator.h"
#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/impl/Operator.h"

namespace execution {

class AggregateOperator : public SingleChildOperator {
private:
  struct Hasher {
    size_t operator()(const std::vector<AnyValue> &keys) const;
  };
  enum class AggregateMode {
    KEY,
    SUM,
  };

  qoperator::AggregateReturnOp _retOp;
  std::vector<AggregateMode> _aggregateModes;
  std::vector<PhysicalColumnType> _outputColumnTypes;

  using AggregateMap =
      std::unordered_map<std::vector<AnyValue>, std::vector<AnyValue>, Hasher>;
  AggregateMap _aggregate;

  // If present, we have drained the child and have started returning the output
  // in blocks.
  std::optional<AggregateMap::const_iterator> _aggregateIter = std::nullopt;
  size_t _rowsSubmitted = 0;

  void drainChild();
  void handleInputBatch(const Batch &batch);
  void handleInputRow(const Batch &batch, size_t row);

public:
  AggregateOperator(OperatorPtr child, qoperator::AggregateReturnOp retOp);

  std::optional<Batch> poll() override;
};

} // namespace execution