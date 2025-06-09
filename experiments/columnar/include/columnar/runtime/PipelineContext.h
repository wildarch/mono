#pragma once

#include <llvm/Support/Allocator.h>
#include <parquet/file_reader.h>

namespace columnar::runtime {

class PipelineContext {
private:
  llvm::BumpPtrAllocator _allocator;

public:
  void *allocate(std::size_t size, std::size_t alignment);
  void reset();
};

} // namespace columnar::runtime
