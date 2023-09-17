#include <arrow/api.h>
#include <iostream>
#include <parquet/file_reader.h>

#include "execution/Batch.h"
#include "execution/ParquetScanner.h"
#include "execution/operator/Operator.h"
#include "execution/operator/ParquetScanOperator.h"

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

  execution::OperatorPtr root =
      std::make_shared<execution::ParquetScanOperator>(*reader, columns);

  int64_t sum = 0;
  std::optional<execution::Batch> batch;
  while ((batch = root->poll())) {
    auto &column = batch->columns().at(0);
    for (auto val : column.get<execution::PhysicalColumnType::INT32>()) {
      sum += val;
    }
  }

  std::cout << "sum: " << sum << "\n";

  return 0;
}