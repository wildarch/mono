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
static int64_t readColumn(parquet::ColumnReader &abstractReader,
                          const Batch &batch, Batch::Column &target) {
  auto &reader = castReader<type>(abstractReader);

  int64_t valuesRead;
  if constexpr (type == PhysicalColumnType::STRING_PTR) {
    throw std::logic_error("Reading string columns is not yet supported");
  } else {
    reader.ReadBatch(batch.rows(), nullptr, nullptr, target.getForWrite<type>(),
                     &valuesRead);
  }
  return valuesRead;
}

static int64_t readColumn(parquet::ColumnReader &abstractReader,
                          const Batch &batch, Batch::Column &target,
                          PhysicalColumnType type) {
  switch (type) {
#define CASE(v)                                                                \
  case PhysicalColumnType::v:                                                  \
    return readColumn<PhysicalColumnType::v>(abstractReader, batch, target);

    CASE(INT32)
    CASE(DOUBLE)
    CASE(STRING_PTR)
#undef CASE
  default:
    __builtin_unreachable();
  }
}

void ParquetScanner::scan(Batch &batch) {
  if (!_currentRowGroup &&
      _currentRowGroupIdx < _reader.metadata()->num_row_groups()) {
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
    // TODO: check number of values read
    valuesRead =
        readColumn(*columnReader, batch, batchColumn, _columnsToRead[i].type);
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
  bool hasMoreRowGroups =
      _currentRowGroupIdx < _reader.metadata()->num_row_groups();
  if (hasMoreRowGroups) {
    return true;
  }
  bool hasRowsLeftInCurrentGroup = _currentRowGroup.has_value();
  if (hasRowsLeftInCurrentGroup) {
    return true;
  }
  return false;
}

} // namespace execution