#include "util/BumpAllocator.h"
#include <cstdint>
#include <iostream>

using namespace dblang;

namespace {

// A small struct with a non-trivial alignment to exercise alignment handling.
struct alignas(16) OverAligned {
  int a;
  int b;
};

int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL: " << #cond << " at line " << __LINE__ << "\n";       \
      failures++;                                                              \
    }                                                                          \
  } while (0)

// Allocate a single object and check it is aligned and constructible.
void testSingleAllocation() {
  BumpAllocator alloc;
  int *p = alloc.create<int>(42);
  CHECK(p != nullptr);
  CHECK(*p == 42);
  CHECK(reinterpret_cast<std::uintptr_t>(p) % alignof(int) == 0);
}

// Allocate many objects; they should not overlap and all be aligned.
void testManyAllocations() {
  BumpAllocator alloc;
  constexpr int N = 1000;
  int *ptrs[N];
  for (int i = 0; i < N; i++) {
    ptrs[i] = alloc.create<int>(i);
    CHECK(reinterpret_cast<std::uintptr_t>(ptrs[i]) % alignof(int) == 0);
  }
  for (int i = 0; i < N; i++) {
    CHECK(*ptrs[i] == i);
  }
}

// A single allocation larger than the default slab should still work.
void testLargeAllocation() {
  BumpAllocator alloc;
  constexpr std::size_t Big = 1 << 20; // 1 MiB, larger than the default slab.
  char *p = static_cast<char *>(alloc.allocate(Big));
  CHECK(p != nullptr);
  // Touch the memory to make sure it is really usable.
  p[0] = 'a';
  p[Big - 1] = 'z';
  CHECK(p[0] == 'a');
  CHECK(p[Big - 1] == 'z');
}

// Over-aligned types must be aligned to their stricter alignment.
void testOverAligned() {
  BumpAllocator alloc;
  OverAligned *p = alloc.create<OverAligned>();
  CHECK(reinterpret_cast<std::uintptr_t>(p) % alignof(OverAligned) == 0);
  p->a = 1;
  p->b = 2;
  CHECK(p->a == 1 && p->b == 2);
}

// allocateArray returns unconstructed storage; elements must be aligned.
void testAllocateArray() {
  BumpAllocator alloc;
  constexpr std::size_t N = 10;
  double *arr = alloc.allocateArray<double>(N);
  CHECK(reinterpret_cast<std::uintptr_t>(arr) % alignof(double) == 0);
  for (std::size_t i = 0; i < N; i++) {
    arr[i] = static_cast<double>(i);
  }
  for (std::size_t i = 0; i < N; i++) {
    CHECK(arr[i] == static_cast<double>(i));
  }
}

// reset() frees all memory; the allocator remains usable afterwards.
void testReset() {
  BumpAllocator alloc;
  alloc.create<int>(1);
  alloc.create<int>(2);
  CHECK(alloc.getTotalMemory() > 0);
  alloc.reset();
  CHECK(alloc.getTotalMemory() == 0);
  // Still usable after reset.
  int *p = alloc.create<int>(7);
  CHECK(*p == 7);
}

// getTotalMemory should grow as slabs are added.
void testTotalMemory() {
  BumpAllocator alloc;
  std::size_t before = alloc.getTotalMemory();
  alloc.create<int>(1);
  std::size_t after = alloc.getTotalMemory();
  CHECK(after >= before);
  CHECK(after > 0);
}

} // namespace

int main() {
  testSingleAllocation();
  testManyAllocations();
  testLargeAllocation();
  testOverAligned();
  testAllocateArray();
  testReset();
  testTotalMemory();

  if (failures) {
    std::cout << failures << " check(s) FAILED\n";
    return 1;
  }
  std::cout << "all BumpAllocator tests passed\n";
  return 0;
}
