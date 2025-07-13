#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include "columnar/runtime/HashTable.h"

namespace columnar::runtime {

class TupleArena {
private:
  /**
   * Size of the tuple, rounded up to meet alignment requirements for the next
   * tuple.
   */
  std::size_t _tupleSizeAligned;

  /** Points to the start of free memory in the current slab. */
  std::byte *_curPtr = nullptr;
  /** Number of tuples we can still allocate to the current slab. */
  std::size_t _curLeft = 0;

  llvm::SmallVector<void *, 4> _slabs;

  void *allocateSlow();

public:
  TupleArena(std::size_t tupleSize, std::size_t tupleAlignment);

  void *allocate() {
    if (_curLeft > 0) [[likely]] {
      _curLeft--;
      std::byte *ptr = _curPtr;
      _curPtr += _tupleSizeAligned;
      return ptr;
    }

    return allocateSlow();
  }
};

class TupleBufferLocal {
private:
  std::array<TupleArena, HashPartitioning::NUM_PARTITIONS> _partitions;

public:
  TupleBufferLocal(std::size_t tupleSize, std::size_t tupleAlignment);

  void insert(llvm::ArrayRef<std::uint64_t> hashes,
              llvm::MutableArrayRef<void *> result) {
    for (auto [i, h] : llvm::enumerate(hashes)) {
      auto partIdx = HashPartitioning::partIdxForHash(h);
      auto &part = _partitions[partIdx];
      void *ptr = part.allocate();

      // Copy the hash into the newly allocated tuple.
      std::memcpy(ptr, &h, sizeof(h));

      // Return the pointer to the tuple.
      result[i] = ptr;
    }
  }
};

} // namespace columnar::runtime
