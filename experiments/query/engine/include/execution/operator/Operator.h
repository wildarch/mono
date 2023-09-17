#pragma once

#include <memory>
#include <optional>

#include "execution/Batch.h"

namespace execution {

enum class OperatorKind {
  PARQUET_SCAN,
};

class Operator;
using OperatorPtr = std::shared_ptr<Operator>;

class Operator {
private:
  OperatorKind _kind;
  std::vector<OperatorPtr> _children;

protected:
  // Use in call to Operator constructor to specify that the operator does not
  // have any children.
  static constexpr std::span<OperatorPtr> NO_CHILDREN =
      std::span<OperatorPtr>();

public:
  inline Operator(OperatorKind kind, std::span<const OperatorPtr> children)
      : _kind(kind), _children(children.begin(), children.end()) {}
  /** An operator with exactly one child. */
  inline Operator(OperatorKind kind, OperatorPtr child)
      : _kind(kind), _children() {
    _children.push_back(child);
  }
  virtual ~Operator() = default;
  virtual std::optional<Batch> poll() = 0;

  auto kind() { return _kind; }
  const auto &children() { return _children; }
};

} // namespace execution