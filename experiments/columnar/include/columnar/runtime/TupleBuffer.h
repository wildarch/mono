#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <llvm-20/llvm/Support/Alignment.h>
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

  void *ptr() const { return _ptr; }

  std::size_t nTuples() const { return _nTuples; }
};

class TupleLayout {
private:
  std::size_t _size;
  std::size_t _alignment;
  std::size_t _sizeAligned;

public:
  TupleLayout(std::size_t size, std::size_t alignment)
      : _size(size), _alignment(alignment),
        _sizeAligned(llvm::alignTo(size, alignment)) {}

  std::size_t size() const { return _size; }

  std::size_t alignment() const { return _alignment; }

  /**
   * Size of the tuple, rounded up to meet alignment requirements for the next
   * tuple.
   */
  std::size_t sizeAligned() const { return _sizeAligned; }
};

class TupleArena {
private:
  TupleLayout _layout;

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
      _curPtr += _layout.sizeAligned();
      return ptr;
    }

    return allocateSlow();
  }

  // Returns total number of tuples in all slabs taken.
  std::size_t takeSlabs(std::vector<OwnedTupleSlab> &slabs);
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
  friend class HashTableBuilder;

private:
  struct Partition {
    std::vector<OwnedTupleSlab> slabs;
  };

  TupleLayout _tupleLayout;
  std::mutex _mutex;
  std::array<Partition, HashPartitioning::NUM_PARTITIONS> _parts;
  std::vector<Allocator> _allocators;
  std::size_t _nTuples = 0;

public:
  TupleBufferGlobal(std::size_t tupleSize, std::size_t tupleAlignment)
      : _tupleLayout(tupleSize, tupleAlignment) {}

  void merge(TupleBufferLocal &local);

  void dump();
};

class HashTableBuilder {
private:
  HashTable &_table;
  const TupleBufferGlobal &_buffer;

  void countPerSlot(std::size_t partIdx);
  void prefixSum(std::size_t partIdx, std::size_t prevCount);
  void copyTuples(std::size_t partIdx);

public:
  HashTableBuilder(HashTable &table, const TupleBufferGlobal &buffer);

  // Allocate large enough directory, tupleStorage and set shift.
  void initializeTable();

  // Count, exclusive sum, and copy.
  void postProcessBuild(std::size_t partIdx, std::size_t prevCount);
};

} // namespace columnar::runtime
