#include <arrow/api.h>
#include <iostream>
#include <parquet/file_reader.h>

#include "execution/Batch.h"
#include "execution/ParquetScanner.h"

constexpr uint32_t ROWS_PER_BATCH = 1024;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Expect path to parquet file for reading" << std::endl;
    return 1;
  }
  auto reader = parquet::ParquetFileReader::OpenFile(argv[1]);

  std::cout << "Reading column: "
            << reader->metadata()->schema()->Column(0)->name() << "\n";
  std::array<execution::ParquetScanner::ColumnToRead, 1> columns{
      execution::ParquetScanner::ColumnToRead{
          .columnId = 0,
          .type = execution::PhysicalColumnType::INT32,
      }};

  execution::ParquetScanner scanner(*reader, columns);

  std::array<execution::PhysicalColumnType, 1> batchColumnTypes{
      execution::PhysicalColumnType::INT32,
  };
  execution::Batch batch(batchColumnTypes, ROWS_PER_BATCH);

  int64_t sum = 0;
  while (scanner.hasNext()) {
    scanner.scan(batch);

    auto &column = batch.columns().at(0);
    for (auto val : column.get<execution::PhysicalColumnType::INT32>()) {
      sum += val;
    }
  }

  std::cout << "sum: " << sum << "\n";

  return 0;
}