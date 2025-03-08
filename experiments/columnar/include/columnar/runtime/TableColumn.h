#pragma once

#include "llvm/Support/FileSystem.h"

namespace columnar::runtime {

class TableColumn {
private:
  llvm::sys::fs::file_t _file = llvm::sys::fs::kInvalidFile;
  llvm::sys::fs::mapped_file_region _mapped;

public:
  ~TableColumn();

  llvm::Error open(llvm::Twine path);
  llvm::Error close();

  void read(std::size_t start, std::size_t size,
            llvm::MutableArrayRef<std::int32_t> dest);
};

} // namespace columnar::runtime