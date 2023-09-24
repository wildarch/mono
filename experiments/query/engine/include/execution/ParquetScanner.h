#pragma once

#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <span>

#include "execution/Batch.h"

namespace execution {

class ParquetScanner {
public:
  struct ColumnToRead {
    int columnId;
    PhysicalColumnType type;
  };

private:
  std::unique_ptr<parquet::ParquetFileReader> _reader;
  std::vector<ColumnToRead> _columnsToRead;

  int _currentRowGroupIdx = 0;
  std::optional<std::shared_ptr<parquet::RowGroupReader>> _currentRowGroup;
  std::vector<std::shared_ptr<parquet::ColumnReader>> _columnReaders;
  int64_t _rowGroupValuesRead = 0;

public:
  ParquetScanner(std::unique_ptr<parquet::ParquetFileReader> reader,
                 std::span<const ColumnToRead> columnsToRead)
      : _reader(std::move(reader)),
        _columnsToRead(columnsToRead.begin(), columnsToRead.end()) {}

  bool hasNext();
  void scan(Batch &batch);
};

} // namespace execution