#include <filesystem>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <parquet/file_reader.h>

#include "columnar/Catalog.h"
#include "columnar/Columnar.h"
#include "columnar/parquet/ParquetToCatalog.h"

llvm::cl::opt<std::string>
    dataDir("data", llvm::cl::desc("Directory containing queryable files"),
            llvm::cl::value_desc("path to directory"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext context;
  context.loadDialect<columnar::ColumnarDialect>();

  columnar::Catalog catalog;
  if (!dataDir.empty()) {
    std::filesystem::path dataDirPath(dataDir.getValue());
    if (!std::filesystem::exists(dataDirPath)) {
      llvm::errs() << "Data directory '" << dataDir << "' does not exist\n";
      return 1;
    }

    if (!std::filesystem::is_directory(dataDirPath)) {
      llvm::errs() << "Data directory '" << dataDir << "' is not a directory\n";
      return 1;
    }

    for (const auto &entry : std::filesystem::directory_iterator(dataDirPath)) {
      if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
        try {
          auto reader = parquet::ParquetFileReader::OpenFile(entry.path());
          auto meta = reader->metadata();
          const auto &schema = *meta->schema();
          columnar::parquet::addToCatalog(&context, catalog,
                                          entry.path().string(), schema);
        } catch (const std::exception &e) {
          llvm::errs() << "WARNING: failed to add parquet file to catalog: '"
                       << entry.path() << "': " << e.what() << "\n";
        }
      }
    }
  }

  catalog.dump();

  return 0;
}

/*
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

*/
