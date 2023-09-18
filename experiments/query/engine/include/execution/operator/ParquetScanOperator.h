#pragma once

#include "execution/Batch.h"
#include "execution/ParquetScanner.h"
#include "execution/operator/Operator.h"
#include <parquet/file_reader.h>

namespace execution {

class ParquetScanOperator : public LeafOperator {
private:
  ParquetScanner _scanner;
  std::vector<PhysicalColumnType> _columnTypes;

public:
  ParquetScanOperator(
      parquet::ParquetFileReader &reader,
      std::span<const ParquetScanner::ColumnToRead> columnsToRead);

  std::optional<Batch> poll() override;
};

} // namespace execution