#include <arrow/api.h>
#include <iostream>
#include <parquet/file_reader.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Expect path to parquet file for reading" << std::endl;
    return 1;
  }
  auto reader = parquet::ParquetFileReader::OpenFile(argv[1]);
  auto meta = reader->metadata();
  std::cout << "Rows: " << meta->num_rows()
            << " (groups: " << meta->num_row_groups() << ")\n";

  return 0;
}