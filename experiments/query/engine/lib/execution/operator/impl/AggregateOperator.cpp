#include "execution/operator/impl/AggregateOperator.h"
#include "execution/expression/ConstantExpression.h"
#include <llvm-17/llvm/ADT/Hashing.h>
#include <llvm-17/llvm/Support/ErrorHandling.h>

namespace execution {

size_t AggregateOperator::Hasher::operator()(
    const std::vector<ConstantValue> &keys) const {
  // TODO: consider skipping collection to a vector.
  llvm::SmallVector<size_t> keyHashes;
  keyHashes.reserve(keys.size());
  for (const auto &key : keys) {
    keyHashes.emplace_back(std::hash<ConstantValue>{}(key));
  }
  return llvm::hash_combine_range(keyHashes.begin(), keyHashes.end());
}

std::optional<Batch> AggregateOperator::poll() {
  auto input = child()->poll();
  if (!input) {
    // need to output the results
    llvm_unreachable("not implemented");
  }

  llvm_unreachable("not implemented");
}

} // namespace execution