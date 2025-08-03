#pragma once

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>

namespace columnar::runtime {

using hash64_t = std::uint64_t;

struct Hash {
  // Taken from DuckDB src/include/duckdb/common/types/hash.hpp
  inline static hash64_t hash(uint64_t x) {
    x ^= x >> 32;
    x *= 0xd6e8feb86659fd93U;
    x ^= x >> 32;
    x *= 0xd6e8feb86659fd93U;
    x ^= x >> 32;
    return x;
  }

  inline static hash64_t hash(std::uint32_t x) {
    return hash(std::uint64_t(x));
  }

  // Taken from DuckDB src/common/vector_operations/vector_hash.cpp
  inline static hash64_t combine(hash64_t a, hash64_t b) {
    a ^= a >> 32;
    a *= 0xd6e8feb86659fd93U;
    return a ^ b;
  }

  static void hash(llvm::ArrayRef<std::uint64_t> value,
                   llvm::ArrayRef<std::size_t> sel,
                   llvm::MutableArrayRef<std::uint64_t> result);
};

} // namespace columnar::runtime
