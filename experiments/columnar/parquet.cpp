#include "llvm/ADT/Sequence.h"
#include <cassert>
#include <cstdint>
#include <iostream>

#include <parquet/api/reader.h>
#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/types.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Invalid args\n";
    return 1;
  }

  std::string path(argv[1]);
  try {
    auto reader = parquet::ParquetFileReader::OpenFile(path);
    auto meta = reader->metadata();
    std::cout << "row groups: " << meta->num_row_groups() << "\n";
    std::cout << "columns: " << meta->num_columns() << "\n";

    for (auto rg : llvm::seq(meta->num_row_groups())) {
      auto group = reader->RowGroup(rg);
      auto keyCol = group->Column(0);
      assert(keyCol->type() == parquet::Type::INT32);
      auto keyI32 = static_cast<parquet::Int32Reader *>(keyCol.get());
      while (keyI32->HasNext()) {
        constexpr size_t BATCH_SIZE = 1024;
        std::int32_t batch[BATCH_SIZE];
        std::int64_t valuesRead;
        auto nRead =
            keyI32->ReadBatch(BATCH_SIZE, nullptr, nullptr, batch, &valuesRead);
        for (auto i : llvm::seq(nRead)) {
          std::cout << "value: " << batch[i] << "\n";
        }
      }
    }
  } catch (const std::exception &e) {
    std::cout << "Parquet error: " << e.what() << "\n";
  }
  // TODO
  return 0;
}
