#pragma once

#include <cassert>
#include <crc32intrin.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/bit.h>
#include <llvm/Support/MathExtras.h>

namespace columnar::ht {

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

struct Tuple {
  std::uint64_t hash;

  std::uint64_t key;

  std::uint64_t val1;
  std::uint64_t val2;
};

static constexpr std::size_t PARTITION_BITS = 4;
static constexpr std::size_t NUM_PARTITIONS = 1 << PARTITION_BITS;
static constexpr std::size_t NUM_THREADS = 8;

class BuildHashTableLocal {
private:
  friend class BuildHashTableGlobal;
  // TODO: Use multi-level slab alloc
  std::vector<Tuple> _parts[NUM_PARTITIONS];
  std::uint64_t _counts[NUM_PARTITIONS];

public:
  void addTuple(std::uint64_t key, std::uint64_t val1, std::uint64_t val2) {
    auto hash = hash64(key, 0, 0);
    auto part = hash >> (64 - PARTITION_BITS);
    assert(part < NUM_PARTITIONS);

    _parts[part].push_back(Tuple{hash, key, val1, val2});
    _counts[part]++;
  }
};

class BuildHashTableGlobal {
private:
  BuildHashTableLocal _local[NUM_THREADS];
  std::vector<Tuple> _mergedParts[NUM_PARTITIONS][NUM_THREADS];
  std::uint64_t _counts[NUM_PARTITIONS];

  std::uint64_t *_directory;
  std::uint64_t _dirShift;
  Tuple *_tupleStorage;

public:
  void mergePartitions() {
    auto &part = _mergedParts[0];
    for (auto t : llvm::seq(NUM_THREADS)) {
      for (auto p : llvm::seq(NUM_PARTITIONS)) {
        _mergedParts[p][t] = std::move(_local[t]._parts[p]);
        _counts[p] += _local[t]._counts[p];
      }
    }
  }

  void allocateFinalStorage() {
    std::uint64_t nTuples = 0;
    for (auto c : _counts) {
      nTuples += c;
    }

    // At least twice as large as stricly necessary to avoid collisions.
    auto minDirSize = nTuples * 2;
    _dirShift = llvm::countl_zero(minDirSize);
    auto dirSize = 1ul << (64 - _dirShift);

    _directory = (std::uint64_t *)malloc(sizeof(std::uint64_t) * dirSize);
    _tupleStorage = (Tuple *)malloc(sizeof(Tuple) * nTuples);
  }

  void postProcessBuild(std::size_t part, std::uint64_t prevCount) {
    // NOTE: We have exclusive access to the part of the directory that stores
    // all tuples in the given partition.

    // 1. Compute the size of the directory per slot, and make the bloom
    //    filters.
    for (auto t : llvm::seq(NUM_THREADS)) {
      for (Tuple tuple : _mergedParts[part][t]) {
        auto slot = tuple.hash >> _dirShift;
        _directory[slot] += sizeof(Tuple) << 16;
        // TODO: compute tag
        // _directory[slot] |= computeTag(tuple.hash);
      }
    }

    // 2. Initialize the directory to be a start index.
    std::uint64_t cur = std::uint64_t(_tupleStorage) + prevCount;
    std::uint64_t k = 64 - _dirShift;
    auto start = (part << k) / NUM_PARTITIONS;
    auto end = ((part + 1) << k) / NUM_PARTITIONS;
    for (auto i = start; i < end; i++) {
      auto val = _directory[i] >> 16;
      auto tag = std::uint16_t(_directory[i]);
      _directory[i] = (cur << 16) | tag;
      cur += val;
    }

    // 3. Write out the individual tuples
    for (auto t : llvm::seq(NUM_THREADS)) {
      for (Tuple tuple : _mergedParts[part][t]) {
        auto slot = tuple.hash >> _dirShift;
        auto *target = (Tuple *)(_directory[slot] >> 16);
        *target = tuple;
        _directory[slot] += (sizeof(Tuple) << 16);
      }
    }
  }

  void postProcessBuild() {
    std::uint64_t prevCount = 0;
    for (auto p : llvm::seq(NUM_PARTITIONS)) {
      postProcessBuild(p, prevCount);
      prevCount += _counts[p];
    }
  }
};

} // namespace columnar::ht
