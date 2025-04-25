#pragma once

#include <memory>

#include <mutex>
#include <parquet/file_reader.h>

namespace columnar::runtime {

class TableScanner {
private:
  std::shared_ptr<parquet::ParquetFileReader> _reader;

  std::mutex _mutex;
  std::int32_t _rowGroup = 0;
  std::int32_t _skip = 0;

public:
  // NOTE: Throws if we fail to open the path.
  void open(const std::string &path);

  struct ClaimedRange {
    std::int32_t rowGroup;
    std::int32_t skip;
    std::int64_t size;
  };

  ClaimedRange claimChunk();
};

} // namespace columnar::runtime
