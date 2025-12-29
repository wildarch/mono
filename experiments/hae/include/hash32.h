#pragma once

#include <cstdint>

// Taken from https://nullprogram.com/blog/2018/07/31/
constexpr uint32_t murmurhash32_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6bU;
  x ^= x >> 13;
  x *= 0xc2b2ae35U;
  x ^= x >> 16;
  return x;
}
