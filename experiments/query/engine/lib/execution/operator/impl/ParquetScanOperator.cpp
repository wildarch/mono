#include "execution/operator/impl/ParquetScanOperator.h"
#include "execution/operator/impl/Operator.h"

namespace execution {

static constexpr uint32_t ROWS_PER_BATCH = 1024;

ParquetScanOperator::ParquetScanOperator(
    std::unique_ptr<parquet::ParquetFileReader> reader,
    std::span<const ParquetScanner::ColumnToRead> columnsToRead)
    : LeafOperator(OperatorKind::PARQUET_SCAN),
      _scanner(std::move(reader), columnsToRead) {
  for (const auto &c : columnsToRead) {
    _columnTypes.push_back(c.type);
  }
}

std::optional<Batch> ParquetScanOperator::poll() {
  if (!_scanner.hasNext()) {
    return std::nullopt;
  }

  Batch batch(_columnTypes, ROWS_PER_BATCH);
  _scanner.scan(batch);
  return batch;
}

} // namespace execution