#include <arrow/api.h>
#include <iostream>
#include <parquet/file_reader.h>
#include <stdexcept>

#include "execution/Batch.h"
#include "execution/Common.h"
#include "execution/ParquetScanner.h"
#include "execution/expression/BinaryOperatorExpression.h"
#include "execution/expression/ColumnExpression.h"
#include "execution/expression/ConstantExpression.h"
#include "execution/operator/FilterOperator.h"
#include "execution/operator/Operator.h"
#include "execution/operator/ParquetScanOperator.h"

constexpr int DAYS_SINCE_EPOCH_1998_09_02 = 10471;

static int getColumn(parquet::ParquetFileReader &reader,
                     std::string_view name) {
  auto schema = reader.metadata()->schema();
  for (int i = 0; i < schema->num_columns(); i++) {
    auto *col = schema->Column(i);
    if (col->name() == name) {
      return i;
    }
  }

  std::cerr << schema->ToString();

  throw std::invalid_argument(std::string("no such column: ") +
                              std::string(name));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Expect path to parquet file for reading" << std::endl;
    return 1;
  }
  auto reader = parquet::ParquetFileReader::OpenFile(argv[1]);

  auto l_lineorderkey = getColumn(*reader, "l_orderkey");
  auto l_shipdate = getColumn(*reader, "l_shipdate");
  std::array<execution::ParquetScanner::ColumnToRead, 2> columns{
      execution::ParquetScanner::ColumnToRead{
          .columnId = l_lineorderkey,
          .type = execution::PhysicalColumnType::INT32,
      },
      execution::ParquetScanner::ColumnToRead{
          .columnId = l_shipdate,
          .type = execution::PhysicalColumnType::INT32,
      },
  };

  execution::OperatorPtr root =
      std::make_shared<execution::ParquetScanOperator>(*reader, columns);
  // Add the filter: l_shipdate <= '1998-09-02'
  auto filterExpr = std::make_shared<execution::BinaryOperatorExpression>(
      std::make_shared<execution::ColumnExpression>(
          execution::ColumnIdx(1), execution::PhysicalColumnType::INT32),
      execution::BinaryOperator::LE_INT32,
      std::make_shared<execution::ConstantExpression>(
          execution::ConstantValue(DAYS_SINCE_EPOCH_1998_09_02)));
  root = std::make_shared<execution::FilterOperator>(root, filterExpr);

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