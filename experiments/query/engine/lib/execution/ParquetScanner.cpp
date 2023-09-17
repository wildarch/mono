#include <parquet/column_reader.h>

#include "execution/Batch.h"
#include "execution/ParquetScanner.h"

namespace execution {

template <PhysicalColumnType type>
auto &castReader(parquet::ColumnReader &reader);

#define CAST_READER_CASE(type, parquet_class)                                  \
  template <>                                                                  \
  auto &castReader<PhysicalColumnType::type>(parquet::ColumnReader & reader) { \
    return static_cast<parquet::parquet_class &>(reader);                      \
  }

CAST_READER_CASE(INT32, Int32Reader)
CAST_READER_CASE(DOUBLE, DoubleReader)
CAST_READER_CASE(STRING_PTR, ByteArrayReader)
#undef CAST_READER_CASE

template <PhysicalColumnType type>
void readColumn(parquet::ColumnReader &abstractReader, Batch::Column &target) {
  auto &reader = castReader<type>(abstractReader);
}

int64_t readColumnTest(parquet::ColumnReader &abstractReader,
                       const Batch &batch, Batch::Column &target) {
  auto &reader = castReader<PhysicalColumnType::INT32>(abstractReader);

  int64_t valuesRead;
  reader.ReadBatch(batch.rows(), nullptr, nullptr,
                   target.getForWrite<PhysicalColumnType::INT32>(),
                   &valuesRead);
  return valuesRead;
}

void readColumn(parquet::ColumnReader &abstractReader, Batch::Column &target,
                PhysicalColumnType type) {
  switch (type) {
#define CASE(v)                                                                \
  case PhysicalColumnType::v:                                                  \
    readColumn<PhysicalColumnType::v>(abstractReader, target);                 \
    break;

    CASE(INT32)
    CASE(DOUBLE)
    CASE(STRING_PTR)
#undef CASE
  }
}

void ParquetScanner::scan(Batch &batch) {
  if (!_currentRowGroup) {
    _currentRowGroup = _reader.RowGroup(_currentRowGroupIdx);
    assert(_currentRowGroup);
    _rowGroupValuesRead = 0;
    auto &rowGroup = **_currentRowGroup;
    _columnReaders.clear();
    for (const auto &column : _columnsToRead) {
      _columnReaders.emplace_back(rowGroup.Column(column.columnId));
    }
  }

  int64_t valuesRead;
  for (size_t i = 0; i < batch.columns().size(); i++) {
    auto &columnReader = _columnReaders[i];
    auto &batchColumn = batch.columnsForWrite().at(i);
    // TODO: check types
    // TODO: check number of values read
    valuesRead = readColumnTest(*columnReader, batch, batchColumn);
  }

  batch.setRows(valuesRead);
  _rowGroupValuesRead += valuesRead;
  auto &rowGroup = **_currentRowGroup;
  if (_rowGroupValuesRead == rowGroup.metadata()->num_rows()) {
    _currentRowGroupIdx++;
    _currentRowGroup = std::nullopt;
  }
}

bool ParquetScanner::hasNext() {
  return _currentRowGroupIdx < _reader.metadata()->num_row_groups();
}

} // namespace execution