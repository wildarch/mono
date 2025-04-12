#include <algorithm>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>

#include "columnar/runtime/TableScanner.h"

namespace columnar::runtime {

static llvm::Expected<std::size_t> readTableSize(llvm::sys::fs::file_t file) {
  llvm::SmallString<256> buffer;
  if (auto err = llvm::sys::fs::readNativeFileToEOF(file, buffer)) {
    return err;
  }

  auto json = llvm::json::parse(buffer);
  if (auto err = json.takeError()) {
    return err;
  }

  auto table = json->getAsObject();
  if (!table) {
    return llvm::createStringError("table meta is not valid JSON");
  }

  auto tableSize = table->getInteger("TableSize");
  if (!tableSize) {
    return llvm::createStringError(
        "table meta does not have 'TableSize' property");
  }

  return *tableSize;
}

llvm::Error TableScanner::open(llvm::Twine path) {
  auto file = llvm::sys::fs::openNativeFileForRead(path);
  if (auto err = file.takeError()) {
    return err;
  }

  auto tableSize = readTableSize(*file);
  if (auto err = tableSize.takeError()) {
    llvm::sys::fs::closeFile(*file);
    return err;
  }

  _tableSize = *tableSize;
  llvm::sys::fs::closeFile(*file);
  return llvm::Error::success();
}

auto TableScanner::claimChunk() -> ClaimedRange {
  constexpr std::size_t CHUNK_SIZE = 1024;
  std::size_t start =
      _nextStart.fetch_add(CHUNK_SIZE, std::memory_order_relaxed);
  if (start > _tableSize) {
    // Nothing left to read.
    return ClaimedRange{0, 0};
  }

  return ClaimedRange{
      .start = start,
      .size = std::min(CHUNK_SIZE, _tableSize - start),
  };
}

} // namespace columnar::runtime
