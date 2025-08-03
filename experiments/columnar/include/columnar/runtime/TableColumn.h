#pragma once

#include <parquet/file_reader.h>

#include "columnar/runtime/ByteArray.h"
#include "columnar/runtime/PipelineContext.h"

namespace columnar::runtime {

class TableColumn {
private:
  int _idx;
  std::shared_ptr<parquet::ParquetFileReader> _reader;

public:
  TableColumn(int idx);

  void open(const std::string &path);
  void close();

  void read(int rowGroup, int skip, std::int64_t size, std::int32_t *buffer);
  void read(PipelineContext &ctx, int rowGroup, int skip, std::int64_t size,
            ByteArray *buffer);
};

} // namespace columnar::runtime
