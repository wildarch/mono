#pragma once

#include <memory>
#include <optional>

#include "execution/Batch.h"

namespace execution {

enum class OperatorKind {
  PARQUET_SCAN,
  FILTER,
  AGGREGATE,
};

class Operator;
using OperatorPtr = std::shared_ptr<Operator>;

class Operator {
private:
  OperatorKind _kind;
  std::vector<OperatorPtr> _children;

protected:
  inline Operator(OperatorKind kind, std::span<const OperatorPtr> children)
      : _kind(kind), _children(children.begin(), children.end()) {}
  virtual ~Operator() = default;

public:
  virtual std::optional<Batch> poll() = 0;

  auto kind() { return _kind; }
  const auto &children() { return _children; }
};

/** Base class for operators without any children */
class LeafOperator : public Operator {
public:
  inline LeafOperator(OperatorKind kind)
      : Operator(kind, std::span<const OperatorPtr>()) {}
};

/** Base class for operators with exactly one child. */
class SingleChildOperator : public Operator {
public:
  inline SingleChildOperator(OperatorKind kind, OperatorPtr child)
      : Operator(kind, std::array<OperatorPtr, 1>{child}) {}

  const auto &child() { return children().at(0); }
};

} // namespace execution