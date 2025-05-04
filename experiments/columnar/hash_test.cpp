#include <crc32intrin.h>
#include <cstdint>
#include <iostream>
#include <vector>

inline std::uint64_t hash32(std::uint32_t key, std::uint32_t seed) {
  std::uint64_t k = 0x8648DBDB;
  std::uint32_t crc = _mm_crc32_u32(seed, key);
  return crc * ((k << 32) + 1);
}

int main(int argc, char **argv) {
  static constexpr std::size_t IDX_BITS = 27;
  static constexpr std::size_t TABLE_SIZE = 1UL << IDX_BITS;
  static constexpr std::size_t N_TO_INSERT = TABLE_SIZE / 2;

  std::vector<int> counts(TABLE_SIZE);
  std::size_t collisions = 0;
  int maxPerSlot = 0;
  for (std::uint32_t i = 0; i < N_TO_INSERT; i++) {
    auto h = hash32(i, 0);
    auto slot = h >> (64 - IDX_BITS);

    if (counts[slot]) {
      collisions++;
    }

    counts[slot]++;
  }

  for (auto c : counts) {
    maxPerSlot = std::max(maxPerSlot, c);
  }

  auto collisionPct = collisions * 100.0 / N_TO_INSERT;
  std::cout << "collisions: " << collisions << " (" << collisionPct << "%)"
            << "\n";
  std::cout << "max per slot: " << maxPerSlot << "\n";

  /*
  for (std::size_t i = 0; i < N; i++) {
    std::cout << i << ": " << counts[i] << "\n";
  }
  */

  /*
  for (auto c : counts) {
    std::cout << c << " ";
  }
  std::cout << "\n";
  */

  return 0;
}
