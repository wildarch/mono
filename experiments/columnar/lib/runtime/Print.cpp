#include <iostream>
#include <sstream>

#include "columnar/runtime/Print.h"

namespace columnar::runtime {

void PrintChunk::append(llvm::ArrayRef<std::int32_t> values,
                        llvm::ArrayRef<std::size_t> sel) {
  assert(values.size() == lines.size());

  for (auto [line, idx] : llvm::enumerate(sel)) {
    lines[line] << values[idx];
  }
}

void Printer::write(PrintChunk &chunk) {
  for (auto &l : chunk.lines) {
    l << "\n";
    // TODO: Avoid string copy here.
    std::cout << l.str();
  }
}

} // namespace columnar::runtime