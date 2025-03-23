#pragma once

#include <atomic>

#include "llvm/Support/Error.h"

namespace columnar::runtime {

class TableScanner {
private:
  std::size_t _tableSize;
  std::atomic_size_t _nextStart = 0;

public:
  llvm::Error open(llvm::Twine path);

  struct ClaimedRange {
    std::size_t start;
    std::size_t size;
  };

  ClaimedRange claimChunk();
};

} // namespace columnar::runtime