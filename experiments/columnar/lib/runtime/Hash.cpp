#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include "columnar/runtime/Hash.h"

namespace columnar::runtime {

void Hash::hash(llvm::ArrayRef<std::uint32_t> value,
                llvm::ArrayRef<std::size_t> sel,
                llvm::MutableArrayRef<std::uint64_t> result) {
  for (auto [i, idx] : llvm::enumerate(sel)) {
    result[i] = hash(value[idx]);
  }
}

} // namespace columnar::runtime
