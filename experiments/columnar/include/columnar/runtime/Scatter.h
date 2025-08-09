#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>

#include "columnar/runtime/Allocator.h"
#include "columnar/runtime/ByteArray.h"

namespace columnar::runtime {

template <typename T>
void scatterPrimitive(llvm::ArrayRef<std::size_t> sel, llvm::ArrayRef<T> value,
                      llvm::ArrayRef<T *> dest) {
  for (auto [i, idx] : llvm::enumerate(sel)) {
    *dest[i] = value[idx];
  }
}

inline void scatterByteArray(llvm::ArrayRef<std::size_t> sel,
                             llvm::ArrayRef<ByteArray> value,
                             llvm::ArrayRef<ByteArray *> dest,
                             Allocator &alloc) {
  for (auto [i, idx] : llvm::enumerate(sel)) {
    *dest[i] = ByteArray(value[idx], alloc);
  }
}

} // namespace columnar::runtime
