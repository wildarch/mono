#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace columnar::runtime {

struct HashPartitioning {
  // Use the 4 most significant bits of the hash for partitioning.
  static constexpr std::size_t PARTITION_IDX_BITS = 4;
  static constexpr std::size_t NUM_PARTITIONS = 1 << PARTITION_IDX_BITS;

  static inline std::size_t partIdxForHash(std::uint64_t hash) {
    constexpr auto hashBits = sizeof(hash) * 8;
    auto res = hash >> (hashBits - PARTITION_IDX_BITS);
    assert(res < NUM_PARTITIONS);
    return res;
  }
};

} // namespace columnar::runtime
