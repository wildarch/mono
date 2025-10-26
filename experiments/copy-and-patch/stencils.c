#include <stdint.h>

#define STENCIL_FUNCTION __attribute__((preserve_none))

extern void cnp_func_hole(void) STENCIL_FUNCTION;

#define STENCIL_HOLE(type) (type)((uintptr_t)&cnp_value_hole)
#define STENCIL_HOLE_INT (int)((uintptr_t)&cnp_int_hole)
#define DECLARE_STENCIL_OUTPUT(...)                                            \
  typedef void (*stencil_output_fn)(__VA_ARGS__) STENCIL_FUNCTION;             \
  stencil_output_fn stencil_output = (stencil_output_fn) & cnp_func_hole;

STENCIL_FUNCTION void load_int_reg1() {
  int a = 0xAABBCCDD;
  DECLARE_STENCIL_OUTPUT(int);
  stencil_output(a);
}

STENCIL_FUNCTION void load_int_reg2(int a) {
  int b = 0xAABBCCDD;
  DECLARE_STENCIL_OUTPUT(int, int);
  stencil_output(a, b);
}

STENCIL_FUNCTION void add_int1_int2(int a, int b) {
  int c = a + b;
  DECLARE_STENCIL_OUTPUT(int);
  stencil_output(c);
}

STENCIL_FUNCTION int return_int1(int a) { return a; }
