#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include "columnar/runtime/Allocator.h"
#include "columnar/runtime/HashTable.h"

namespace columnar::runtime {

class OwnedTupleSlab {
private:
  void *_ptr;
  std::size_t _nTuples;

public:
  OwnedTupleSlab(void *ptr, std::size_t nTuples)
      : _ptr(ptr), _nTuples(nTuples) {}
  ~OwnedTupleSlab() { std::free(_ptr); }
};

class TupleArena {
private:
  /**
   * Size of the tuple, rounded up to meet alignment requirements for the next
   * tuple.
   */
  std::size_t _tupleSizeAligned;
  std::size_t _tupleAlignment;

  /** Points to the start of free memory in the current slab. */
  std::byte *_curPtr = nullptr;
  /** Number of tuples we can still allocate to the current slab. */
  std::size_t _curLeft = 0;

  llvm::SmallVector<void *, 4> _slabs;

  static std::size_t slabSize(std::size_t n);

  void *allocateSlow();

public:
  TupleArena(std::size_t tupleSize, std::size_t tupleAlignment);
  ~TupleArena();

  void *allocate() {
    if (_curLeft > 0) [[likely]] {
      _curLeft--;
      std::byte *ptr = _curPtr;
      _curPtr += _tupleSizeAligned;
      return ptr;
    }

    return allocateSlow();
  }

  void takeSlabs(std::vector<OwnedTupleSlab> &slabs);
};

class TupleBufferLocal {
  friend class TupleBufferGlobal;

private:
  std::array<TupleArena, HashPartitioning::NUM_PARTITIONS> _partitions;
  Allocator _allocator;

public:
  TupleBufferLocal(std::size_t tupleSize, std::size_t tupleAlignment);

  void insert(llvm::ArrayRef<std::uint64_t> hashes,
              llvm::MutableArrayRef<void *> result);

  Allocator *allocator() { return &_allocator; }
};

class TupleBufferGlobal {
private:
  struct Partition {
    std::vector<OwnedTupleSlab> _slabs;
  };

  std::mutex _mutex;
  std::array<Partition, HashPartitioning::NUM_PARTITIONS> _parts;
  std::vector<Allocator> _allocators;

public:
  void merge(TupleBufferLocal &local);
};

} // namespace columnar::runtime
