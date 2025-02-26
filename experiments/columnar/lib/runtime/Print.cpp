#include <iostream>

#include "columnar/runtime/Print.h"

namespace columnar::runtime {

void PrintChunk::append(llvm::ArrayRef<std::int32_t> values) {
  assert(values.size() == lines.size());
  for (auto [i, v] : llvm::enumerate(values)) {
    lines[i] += v;
  }
}

void Printer::write(PrintChunk &chunk) {
  for (auto &l : chunk.lines) {
    l += "\n";
    std::cout << l;
  }
}

} // namespace columnar::runtime