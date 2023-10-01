#include "execution/operator/impl/ProjectOperator.h"

namespace execution {

std::optional<Batch> ProjectOperator::poll() {
  auto input = child()->poll();
  if (!input) {
    return std::nullopt;
  }

  llvm_unreachable("not implemented");
}

} // namespace execution