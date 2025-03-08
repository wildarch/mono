#pragma once

#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"

namespace columnar::runtime {

struct PrintChunk {
  std::vector<std::string> lines;

  inline PrintChunk(std::size_t size) : lines(size) {}

  void append(llvm::ArrayRef<std::int32_t> values,
              llvm::ArrayRef<std::uint32_t> sel);
};

class Printer {
public:
  void write(PrintChunk &chunk);
};

} // namespace columnar::runtime