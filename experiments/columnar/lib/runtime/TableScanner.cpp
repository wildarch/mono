#include <algorithm>

#include "columnar/runtime/TableScanner.h"

namespace columnar::runtime {

TableScanner::TableScanner(std::size_t tableSize) : _tableSize(tableSize) {}

auto TableScanner::claimChunk() -> ClaimedRange {
  constexpr std::size_t CHUNK_SIZE = 1024;
  std::size_t start =
      _nextStart.fetch_add(CHUNK_SIZE, std::memory_order_relaxed);
  if (start > _tableSize) {
    // Nothing left to read.
    return ClaimedRange{0, 0};
  }

  return ClaimedRange{
      .start = start,
      .size = std::min(CHUNK_SIZE, _tableSize - start),
  };
}

} // namespace columnar::runtime