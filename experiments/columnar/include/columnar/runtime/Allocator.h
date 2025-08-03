#pragma once

#include <llvm/Support/Allocator.h>

namespace columnar::runtime {

class Allocator {
private:
  llvm::BumpPtrAllocator _alloc;

public:
  void *allocate(std::size_t size, std::size_t alignment) {
    return _alloc.Allocate(size, alignment);
  }

  void reset() { _alloc.Reset(); }
};

} // namespace columnar::runtime
