#include <iostream>

#include "columnar/runtime/Print.h"

namespace columnar::runtime {

void PrintChunk::append(llvm::ArrayRef<std::int32_t> values,
                        llvm::ArrayRef<std::uint32_t> sel) {
  assert(values.size() == lines.size());
  for (auto [line, idx] : llvm::enumerate(sel)) {
    lines[line] += values[idx];
  }
}

void Printer::write(PrintChunk &chunk) {
  for (auto &l : chunk.lines) {
    l += "\n";
    std::cout << l;
  }
}

} // namespace columnar::runtime