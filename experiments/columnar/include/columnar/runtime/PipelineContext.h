#pragma once

#include <llvm/Support/Allocator.h>
#include <parquet/file_reader.h>

#include "columnar/runtime/Allocator.h"

namespace columnar::runtime {

class PipelineContext {
private:
  Allocator _allocator;

public:
  auto &allocator() { return _allocator; }

  void reset();
};

} // namespace columnar::runtime
