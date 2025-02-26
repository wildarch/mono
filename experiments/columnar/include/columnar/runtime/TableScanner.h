#pragma once

#include <atomic>

namespace columnar::runtime {

class TableScanner {
private:
  std::size_t _tableSize;
  std::atomic_size_t _nextStart = 0;

public:
  TableScanner(std::size_t tableSize);

  struct ClaimedRange {
    std::size_t start;
    std::size_t size;
  };

  ClaimedRange claimChunk();
};

} // namespace columnar::runtime