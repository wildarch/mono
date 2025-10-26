#include <stddef.h>
#include <stdint.h>
#include <string.h>

uint8_t cnp_stencil_load_int_reg1_code[] = {
    0x41, 0xbc, 0x00, 0x00, 0x00, 0x00, // mov r12d,0x0
};
uint8_t *cnp_copy_load_int_reg1(uint8_t *stencil_start) {
  const size_t stencil_size = sizeof(cnp_stencil_load_int_reg1_code);
  memcpy(stencil_start, cnp_stencil_load_int_reg1_code, stencil_size);
  return stencil_start + stencil_size;
}
void cnp_patch_load_int_reg1(uint8_t *stencil_start, int value) {
  // 2: R_X86_64_32 cnp_value_hole  ->  0x02 offset
  memcpy(stencil_start + 0x2, &value, sizeof(value));
}

uint8_t cnp_stencil_load_int_reg2_code[] = {
    0x41, 0xbd, 0x00, 0x00, 0x00, 0x00, // mov r13d,0x0
};
uint8_t *cnp_copy_load_int_reg2(uint8_t *stencil_start) {
  const size_t stencil_size = sizeof(cnp_stencil_load_int_reg2_code);
  memcpy(stencil_start, cnp_stencil_load_int_reg2_code, stencil_size);
  return stencil_start + stencil_size;
}
void cnp_patch_load_int_reg2(uint8_t *stencil_start, int value) {
  // 12: R_X86_64_32 cnp_value_hole  ->  0x12 - 0x10 base = 0x2
  memcpy(stencil_start + 0x2, &value, sizeof(value));
}

uint8_t cnp_stencil_add_int1_int2_code[] = {
    0x45, 0x01, 0xec, // add r12d,r13d
};
uint8_t *cnp_copy_add_int1_int2(uint8_t *stencil_start) {
  const size_t stencil_size = sizeof(cnp_stencil_add_int1_int2_code);
  memcpy(stencil_start, cnp_stencil_add_int1_int2_code, stencil_size);
  return stencil_start + stencil_size;
}
// No patching needed

uint8_t cnp_stencil_return_int1_code[] = {
    0x44, 0x89, 0xe0, // mov eax,r12d
    0xc3,             // ret
};
uint8_t *cnp_copy_return_int1(uint8_t *stencil_start) {
  const size_t stencil_size = sizeof(cnp_stencil_return_int1_code);
  memcpy(stencil_start, cnp_stencil_return_int1_code, stencil_size);
  return stencil_start + stencil_size;
}
// No patching needed
