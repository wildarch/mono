#include <stdint.h>

#define STENCIL_FUNCTION __attribute__((preserve_none))

extern void cnp_stencil_output(void) STENCIL_FUNCTION;

#define STENCIL_HOLE32(ordinal, type)                                          \
  (type)((uintptr_t)&cnp_small_value_hole_##ordinal)
#define STENCIL_HOLE64(ordinal, type)                                          \
  (type)((uintptr_t)&cnp_large_value_hole_##ordinal)
#define STENCIL_FN_NEAR(ordinal, type) (type) & cnp_near_func_hole_##ordinal
#define STENCIL_FN_FAR(ordinal, type)                                          \
  ({                                                                           \
    uint64_t _cnp_addr_as_int =                                                \
        (uint64_t)((uintptr_t)&cnp_far_func_hole_##ordinal);                   \
    asm volatile("" : "+r"(_cnp_addr_as_int) : : "memory");                    \
    (type) _cnp_addr_as_int;                                                   \
  })
#define DECLARE_STENCIL_OUTPUT(...)                                            \
  typedef void (*stencil_output_fn)(__VA_ARGS__) STENCIL_FUNCTION;             \
  stencil_output_fn stencil_output = (stencil_output_fn) & cnp_stencil_output;

#define DECLARE_EXTERN_HOLES(ordinal)                                          \
  extern char cnp_large_value_hole_##ordinal[100000];                          \
  extern char cnp_small_value_hole_##ordinal[8];                               \
  extern void cnp_near_func_hole_##ordinal(void) STENCIL_FUNCTION;             \
  extern char cnp_far_func_hole_##ordinal[100000];
