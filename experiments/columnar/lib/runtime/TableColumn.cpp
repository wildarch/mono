#include <cassert>

#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/types.h>

#include "columnar/runtime/TableColumn.h"

namespace columnar::runtime {

TableColumn::TableColumn(int idx) : _idx(idx) {}

void TableColumn::open(const std::string &path) {
  _reader = parquet::ParquetFileReader::OpenFile(path);
}

void TableColumn::close() { _reader->Close(); }

void TableColumn::read(int rowGroup, int skip, std::int64_t size,
                       std::int32_t *buffer) {
  // TODO: avoid such reads in IR generation.
  if (size == 0) {
    return;
  }

  auto groupReader = _reader->RowGroup(rowGroup);
  auto colReader = groupReader->Column(_idx);

  assert(colReader->type() == parquet::Type::INT32);
  auto *i32Reader = static_cast<parquet::Int32Reader *>(colReader.get());

  i32Reader->Skip(skip);

  std::int64_t valuesRead;
  i32Reader->ReadBatch(size, nullptr, nullptr, buffer, &valuesRead);
  assert(valuesRead == size);
}

} // namespace columnar::runtime
