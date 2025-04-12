#include <llvm/ADT/Twine.h>

#include "columnar/runtime/TableColumn.h"

namespace columnar::runtime {

TableColumn::~TableColumn() {
  if (_file != llvm::sys::fs::kInvalidFile) {
    _mapped.unmap();
    llvm::sys::fs::closeFile(_file);
  }
}

llvm::Error TableColumn::open(llvm::Twine path) {
  assert(_file == llvm::sys::fs::kInvalidFile);
  auto file = llvm::sys::fs::openNativeFileForRead(path);
  if (auto err = file.takeError()) {
    return err;
  }

  _file = *file;

  llvm::sys::fs::file_status fstat;
  auto ec = llvm::sys::fs::status(_file, fstat);
  if (ec) {
    return llvm::createFileError(path, ec);
  }

  std::uint64_t offset = 0;
  _mapped = llvm::sys::fs::mapped_file_region(
      _file, llvm::sys::fs::mapped_file_region::readonly, fstat.getSize(),
      offset, ec);
  if (ec) {
    return llvm::createFileError(path, ec);
  }

  return llvm::Error::success();
}

llvm::Error TableColumn::close() {
  _mapped.unmap();
  auto ec = llvm::sys::fs::closeFile(_file);
  return llvm::errorCodeToError(ec);
}

void TableColumn::read(std::size_t start, std::size_t size,
                       llvm::MutableArrayRef<std::int32_t> dest) {
  assert(dest.size() >= size);
  std::memcpy(dest.data(), _mapped.const_data() + start * sizeof(std::int32_t),
              size * sizeof(std::int32_t));
}

} // namespace columnar::runtime
