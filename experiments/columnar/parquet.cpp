#include <iostream>
#include <memory>

#include <llvm/ADT/Sequence.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <parquet/column_reader.h>
#include <parquet/file_reader.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Invalid args\n";
    return 1;
  }

  std::string path(argv[1]);

  auto reader = parquet::ParquetFileReader::OpenFile(path);
  auto meta = reader->metadata();
  std::cout << "row groups: " << meta->num_row_groups() << "\n";
  std::cout << "columns: " << meta->num_columns() << "\n";

  /*
  for (auto rg : llvm::seq(meta->num_row_groups())) {
    auto group = reader->RowGroup(rg);
    auto col = group->Column(0);

    auto *colI32 = static_cast<::parquet::Int32Reader *>(col.get());
    colI32->Skip(42);

    auto *colBool = static_cast<::parquet::BoolReader *>(col.get());
    colBool->Skip(42);

    auto *colFloat = static_cast<::parquet::BoolReader *>(col.get());
    colFloat->Skip(42);

    auto *colString = static_cast<::parquet::ByteArrayReader *>(col.get());
    colString->Skip(42);

    group->metadata()->num_rows();
  }
  */

  // TODO
  return 0;
}
