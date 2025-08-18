#include <cstdint>
#include <cstdlib>
#include <mutex>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_ostream.h>

#include "columnar/runtime/Hash.h"
#include "columnar/runtime/TupleBuffer.h"

namespace columnar::runtime {

std::size_t TupleArena::slabSize(std::size_t n) { return 64 << n; }

void *TupleArena::allocateSlow() {
  // Add new slab. Everyt time we add a new slab, we increase the number of
  // elements that can fit in it.
  auto newSlabSize = slabSize(_slabs.size());
  auto *slab = static_cast<std::byte *>(std::aligned_alloc(
      _layout.alignment(), newSlabSize * _layout.sizeAligned()));

  _curPtr = slab;
  _curLeft = newSlabSize;
  _slabs.push_back(slab);

  return allocate();
}

TupleArena::TupleArena(std::size_t tupleSize, std::size_t tupleAlignment)
    : _layout(tupleSize, tupleAlignment) {}

TupleArena::~TupleArena() {
  for (auto *slab : _slabs) {
    std::free(slab);
  }
}

std::size_t TupleArena::takeSlabs(std::vector<OwnedTupleSlab> &slabs) {
  std::size_t totalTuples = 0;
  for (const auto &[i, p] : llvm::enumerate(_slabs)) {
    auto nTuples = slabSize(i);
    // The last (and therefore current) slab.
    if (i == _slabs.size() - 1) {
      nTuples -= _curLeft;
    }

    totalTuples += nTuples;
    slabs.emplace_back(p, nTuples);
  }

  _curPtr = nullptr;
  _curLeft = 0;
  _slabs.clear();
  return totalTuples;
}

TupleBufferLocal::TupleBufferLocal(std::size_t tupleSize,
                                   std::size_t tupleAlignment)
    : _partitions{TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment)} {}

void TupleBufferLocal::insert(llvm::ArrayRef<hash64_t> hashes,
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

void TupleBufferGlobal::merge(TupleBufferLocal &local) {
  std::lock_guard guard(_mutex);
  for (auto [i, part] : llvm::enumerate(local._partitions)) {
    _nTuples += part.takeSlabs(_parts[i].slabs);
  }

  _allocators.push_back(std::move(local._allocator));
}

void TupleBufferGlobal::dump() {
  // NOTE: When debugging, change this to the expected tuple struct type.
  struct Tuple {
    std::uint64_t hash;
    std::uint32_t regionKey;
    char *regionName;
  };

  llvm::errs() << "DUMP of tuple buffer parts=" << _parts.size() << "\n";
  for (const auto &[partIdx, part] : llvm::enumerate(_parts)) {
    llvm::errs() << "part " << partIdx << " slabs=" << part.slabs.size()
                 << "\n";

    for (const auto &[slabIdx, slab] : llvm::enumerate(part.slabs)) {
      llvm::errs() << "slab " << slabIdx << " ptr=" << slab.ptr()
                   << " nTuples=" << slab.nTuples() << "\n";
      const auto *ptr = reinterpret_cast<const Tuple *>(slab.ptr());
      for (std::size_t tupleIdx = 0; tupleIdx < slab.nTuples(); tupleIdx++) {
        llvm::errs() << "tuple ptr: " << ptr + tupleIdx << "\n";
        const auto &tuple = ptr[tupleIdx];
        llvm::errs() << "regionName ptr: " << &tuple.regionName << "\n";
        llvm::errs() << "tuple: hash=" << tuple.hash
                     << " regionKey=" << tuple.regionKey
                     << " regionName=" << tuple.regionName << "\n";
      }
    }
  }
}

void HashTableBuilder::countPerSlot(std::size_t partIdx) {
  auto tupleSize = _buffer._tupleLayout.sizeAligned();
  auto shift = _table._shift;

  for (const auto &slab : _buffer._parts[partIdx].slabs) {
    const auto *begin = static_cast<const std::byte *>(slab.ptr());
    const auto *end = begin + slab.nTuples() * tupleSize;
    for (const auto *ptr = begin; ptr < end; end += tupleSize) {
      std::uint64_t hash;
      std::memcpy(&hash, ptr, sizeof(hash));

      auto slot = hash >> shift;

      // Add bytes for this tuple.
      _table._directory[slot] += tupleSize;
    }
  }
}

void HashTableBuilder::prefixSum(std::size_t partIdx, std::size_t prevCount) {
  // TODO
  std::uintptr_t cur = std::uintptr_t(_table._tupleStorage) + prevCount;

  // Select the slice of the directory used for this partition.
  auto k = 64 - _table._shift;
  auto start = (partIdx << k) / HashPartitioning::NUM_PARTITIONS;
  auto end = ((partIdx + 1) << k) / HashPartitioning::NUM_PARTITIONS;

  auto &directory = _table._directory;
  for (auto i = start; i < end; i++) {
    auto val = directory[i];
    directory[i] = cur;
    cur += val;
  }
}

void HashTableBuilder::copyTuples(std::size_t partIdx) {
  auto tupleSize = _buffer._tupleLayout.sizeAligned();
  auto shift = _table._shift;
  auto &directory = _table._directory;

  for (const auto &slab : _buffer._parts[partIdx].slabs) {
    const auto *begin = static_cast<const std::byte *>(slab.ptr());
    const auto *end = begin + slab.nTuples() * tupleSize;
    for (const auto *ptr = begin; ptr < end; end += tupleSize) {
      std::uint64_t hash;
      std::memcpy(&hash, ptr, sizeof(hash));

      auto slot = hash >> shift;

      // Copy the tuple
      void *target = reinterpret_cast<void *>(directory[slot]);
      std::memcpy(target, ptr, tupleSize);

      // Next tuple for this slot comes after the current tuple.
      directory[slot] += tupleSize;
    }
  }
}

void HashTableBuilder::initializeTable() {
  auto nBuckets = llvm::PowerOf2Ceil(_buffer._nTuples);

  // Set shift
  _table._shift = llvm::countl_zero(nBuckets) + 1;

  // Initialize directory
  _table._directory.clear();
  // NOTE: Initialize all directory entries to 0
  _table._directory.resize(nBuckets);

  // Allocate tuple storage
  std::aligned_alloc(_buffer._tupleLayout.alignment(),
                     _buffer._tupleLayout.sizeAligned() * nBuckets);
}

void HashTableBuilder::postProcessBuild(std::size_t partIdx,
                                        std::size_t prevCount) {
  countPerSlot(partIdx);
  prefixSum(partIdx, prevCount);
  copyTuples(partIdx);
}

} // namespace columnar::runtime
