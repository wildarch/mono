#include "columnar/runtime/PipelineContext.h"
namespace columnar::runtime {

void *PipelineContext::allocate(std::size_t size, std::size_t alignment) {
  return _allocator.Allocate(size, alignment);
}

void PipelineContext::reset() { _allocator.Reset(); }

} // namespace columnar::runtime
