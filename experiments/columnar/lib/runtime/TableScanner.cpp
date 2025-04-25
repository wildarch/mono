#include <mutex>
#include <parquet/file_reader.h>

#include "columnar/runtime/TableScanner.h"

namespace columnar::runtime {

void TableScanner::open(const std::string &path) {
  _reader = parquet::ParquetFileReader::OpenFile(path);
}

auto TableScanner::claimChunk() -> ClaimedRange {
  constexpr std::int64_t CHUNK_SIZE = 1024;

  std::lock_guard guard(_mutex);
  if (_rowGroup >= _reader->metadata()->num_row_groups()) {
    // No more row groups.
    return ClaimedRange{_rowGroup, 0, 0};
  }

  auto rowGroup = _reader->RowGroup(_rowGroup);
  auto leftInGroup = rowGroup->metadata()->num_rows() - _skip;
  auto size = std::min(leftInGroup, CHUNK_SIZE);

  ClaimedRange range{_rowGroup, _skip, size};

  // Update for next read
  _skip += size;
  if (_skip == rowGroup->metadata()->num_rows()) {
    // Exhausted the current group.
    _rowGroup++;
    _skip = 0;
  }

  return range;
}

} // namespace columnar::runtime
