#include "cnp_stencils.h"

// Declare up to the maximum number of holes you need of one type
// in a function:
DECLARE_EXTERN_HOLES(1);
DECLARE_EXTERN_HOLES(2);

STENCIL_FUNCTION
void fused_multiply_add_sqrt_ifnotzero() {
  uint32_t a = STENCIL_HOLE32(1, uint32_t);
  uint32_t b = STENCIL_HOLE32(2, int32_t);
  uint64_t c = STENCIL_HOLE64(1, uint64_t);

  uint64_t fma = a * b + c;

  if (fma == 0) {
    void (*div_trap)(void) = STENCIL_FN_NEAR(1, void (*)(void));
    div_trap();
  }

  uint64_t (*sqrt)(uint64_t) = STENCIL_FN_FAR(1, uint64_t (*)(uint64_t));
  uint64_t result = sqrt(c);

  DECLARE_STENCIL_OUTPUT(uint64_t);
  stencil_output(result);
}
