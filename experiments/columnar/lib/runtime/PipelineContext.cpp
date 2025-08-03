#include "columnar/runtime/PipelineContext.h"

namespace columnar::runtime {

void PipelineContext::reset() { _allocator.reset(); }

} // namespace columnar::runtime
