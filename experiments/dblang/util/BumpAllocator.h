#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace dblang {

/**
 * A bump (arena) allocator that hands out memory from a set of slabs.
 *
 * Memory is allocated by bumping a pointer down through a slab, so individual
 * allocations are never freed; the whole arena is reclaimed at once when the
 * allocator is destroyed or \c reset is called. This makes allocation
 * extremely cheap (a pointer decrement and a bounds check) at the cost of
 * never releasing memory until the arena goes away.
 *
 * This is a "bump down" allocator: the bump pointer starts at the top of each
 * slab and moves downward as allocations are made.
 *
 * The allocator is not thread-safe; callers must provide their own
 * synchronization if used from multiple threads.
 */
class BumpAllocator {
public:
  /// Default slab size, in bytes.
  static constexpr std::size_t DEFAULT_SLAB_SIZE = 4096;

  /// Default alignment for allocations, in bytes.
  static constexpr std::size_t DEFAULT_ALIGNMENT = 8;

  BumpAllocator() = default;

  /// Non-copyable: the allocator owns its slabs.
  BumpAllocator(const BumpAllocator &) = delete;
  BumpAllocator &operator=(const BumpAllocator &) = delete;

  ~BumpAllocator() { deallocateSlabs(); }

  /// Allocate \p size bytes aligned to \p align.
  ///
  /// \p align must be a power of two. The returned pointer is valid until the
  /// allocator is destroyed or \c reset() is called.
  void *allocate(std::size_t size, std::size_t align = DEFAULT_ALIGNMENT) {
    // Round the requested size up to the alignment so that the next
    // allocation stays aligned.
    size = alignTo(size, align);

    // The bump pointer moves down; if the current slab can't satisfy the
    // request, start a fresh one.
    if (size > currentSlabRemaining()) {
      startSlab(size);
    }

    _currentPtr -= size;
    return _currentPtr;
  }

  /// Allocate and default-construct an object of type \c T.
  template <typename T> T *allocate() {
    return new (allocate(sizeof(T), alignof(T))) T();
  }

  /// Allocate and construct an object of type \c T with the given arguments.
  template <typename T, typename... Args> T *create(Args &&...args) {
    return new (allocate(sizeof(T), alignof(T))) T(std::forward<Args>(args)...);
  }

  /// Allocate storage for \a n objects of type \c T without constructing them.
  template <typename T> T *allocateArray(std::size_t n) {
    return static_cast<T *>(allocate(n * sizeof(T), alignof(T)));
  }

  /// Release all memory held by the allocator. All previously returned
  /// pointers become invalid.
  void reset() {
    deallocateSlabs();
    _slabs = nullptr;
    _currentPtr = nullptr;
    _currentEnd = nullptr;
  }

  /// Total number of bytes currently allocated across all slabs.
  std::size_t getTotalMemory() const {
    std::size_t total = 0;
    for (const Slab *s = _slabs; s != nullptr; s = s->next) {
      total += s->size;
    }
    return total;
  }

private:
  struct Slab {
    Slab *next;
    std::size_t size;
    // The slab's storage begins immediately after this header.
  };

  /// The current slab's usable storage begins right after its header.
  static std::size_t slabHeaderSize() { return sizeof(Slab); }

  /// Round \a n up to the next multiple of \a align (a power of two).
  static std::size_t alignTo(std::size_t n, std::size_t align) {
    return (n + align - 1) & ~(align - 1);
  }

  /// Bytes remaining in the current slab, or 0 if there is no current slab.
  std::size_t currentSlabRemaining() const {
    if (_currentPtr == nullptr) {
      return 0;
    }
    return static_cast<std::size_t>(_currentEnd - _currentPtr);
  }

  /// Allocate a new slab large enough for at least \a minSize bytes of
  /// payload and make it the current slab.
  void startSlab(std::size_t minSize) {
    std::size_t size = std::max(DEFAULT_SLAB_SIZE, minSize + slabHeaderSize());
    void *mem = std::malloc(size);
    if (mem == nullptr) {
      throw std::bad_alloc();
    }

    auto *slab = static_cast<Slab *>(mem);
    slab->next = _slabs;
    slab->size = size;
    _slabs = slab;

    // The bump pointer starts at the top of the payload and moves down.
    _currentEnd = reinterpret_cast<char *>(slab) + size;
    _currentPtr = _currentEnd;
  }

  /// Free all slabs.
  void deallocateSlabs() {
    Slab *s = _slabs;
    while (s != nullptr) {
      Slab *next = s->next;
      std::free(s);
      s = next;
    }
  }

  Slab *_slabs = nullptr;
  char *_currentPtr = nullptr;
  char *_currentEnd = nullptr;
};

} // namespace dblang