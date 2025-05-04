#pragma once

#include <cassert>
#include <crc32intrin.h>
#include <cstdint>
#include <limits>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <vector>

namespace columnar::ht {

struct Tuple {
  std::uint64_t hash;

  std::uint64_t key;

  std::uint64_t val1;
  std::uint64_t val2;
};

inline std::uint64_t hash32(std::uint32_t key, std::uint32_t seed) {
  std::uint64_t k = 0x8648DBDB;
  std::uint32_t crc = _mm_crc32_u32(seed, key);
  return crc * ((k << 32) + 1);
}

inline std::uint64_t hash64(std::uint64_t key, std::uint32_t seed1,
                            std::uint32_t seed2) {
  std::uint64_t k = 0x2545F4914F6CDD1D;
  std::uint32_t crc1 = _mm_crc32_u32(seed1, key);
  std::uint32_t crc2 = _mm_crc32_u32(seed2, key);
  std::uint64_t upper = std::uint64_t(crc2) << 32;
  std::uint64_t combined = crc1 | upper;
  return combined * k;
}

class TupleCollect {
private:
  static constexpr std::size_t NUM_PARTITIONS = 16;
  static constexpr std::size_t NUM_PARTITIONS_LOG2 = 4;
  static_assert(std::numeric_limits<std::uint64_t>::max() >>
                    (64 - NUM_PARTITIONS_LOG2) ==
                NUM_PARTITIONS - 1);

  // TODO: Use multi-level slab alloc
  std::vector<Tuple> _parts[NUM_PARTITIONS];
  std::uint64_t _counts[NUM_PARTITIONS];

  std::uint64_t *_directory;
  std::uint64_t _dirShift;
  Tuple *_tupleStorage;

public:
  void addTuple(std::uint64_t key, std::uint64_t val1, std::uint64_t val2) {
    auto hash = hash64(key, 0, 0);
    auto part = hash >> (64 - NUM_PARTITIONS_LOG2);
    assert(part < NUM_PARTITIONS);

    _parts[part].push_back(Tuple{hash, key, val1, val2});
    _counts[part]++;
  }

  static void mergePartitions(llvm::ArrayRef<TupleCollect *> perThread,
                              std::vector<Tuple> **out,
                              std::uint64_t counts[]) {
    auto nParts = NUM_PARTITIONS;
    auto nThreads = perThread.size();

    for (auto t : llvm::seq(nThreads)) {
      for (auto p : llvm::seq(nParts)) {
        out[p][t] = std::move(perThread[t]->_parts[p]);
        counts[p] += perThread[t]->_counts[p];
      }
    }
  }

  void postProcessBuild(std::uint64_t numThreads, std::vector<Tuple> **perPart,
                        std::uint64_t part, std::uint64_t prevCount) {
    // NOTE: We have exclusive access to the part of the directory that stores
    // all tuples in the given partition.

    // 1. Compute the size of the directory per slot, and make the bloom
    //    filters.
    for (auto t : llvm::seq(numThreads)) {
      for (Tuple tuple : perPart[part][t]) {
        auto slot = tuple.hash >> _dirShift;
        _directory[slot] += sizeof(Tuple) << 16;
        // TODO: compute tag
        // _directory[slot] |= computeTag(tuple.hash);
      }
    }

    // 2. Initialize the directory to be a start index.
    std::uint64_t cur = std::uint64_t(_tupleStorage) + prevCount;
    std::uint64_t k = 64 - _dirShift;
    // TODO: Check that this makes sense.
    auto start = (part << k) / NUM_PARTITIONS;
    auto end = ((part + 1) << k) / NUM_PARTITIONS;
    for (auto i = start; i < end; i++) {
      auto val = _directory[i] >> 16;
      auto tag = std::uint16_t(_directory[i]);
      _directory[i] = (cur << 16) | tag;
      cur += val;
    }

    // 3. Write out the individual tuples
    for (auto t : llvm::seq(numThreads)) {
      for (Tuple tuple : perPart[part][t]) {
        auto slot = tuple.hash >> _dirShift;
        auto *target = (Tuple *)(_directory[slot] >> 16);
        *target = tuple;
        _directory[slot] += (sizeof(Tuple) << 16);
      }
    }
  }
};

} // namespace columnar::ht
