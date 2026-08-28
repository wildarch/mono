#pragma once

#include "ir/StringPool.h"
#include "ir/Type.h"
#include "ir/TypeImpl.h"
#include "util/BumpAllocator.h"

#include <cstddef>
#include <span>
#include <unordered_set>

namespace dblang::ir {

struct Def;

/**
 * Stores unique type implementations.
 *
 * Types are interned: each distinct type (compared structurally) is stored
 * exactly once, in memory owned by an internal \c BumpAllocator. Callers
 * receive a \c Type wrapper whose underlying \c TypeImpl pointer is stable for
 * the lifetime of the pool, so two \c Type values can be compared for
 * equivalence cheaply (a pointer comparison).
 *
 * The pool is not thread-safe; callers must provide their own synchronization
 * if used from multiple threads.
 */
class TypePool {
public:
  TypePool() = default;

  Type getPrimitive(impl::TypeKind kind);
  Type getUnresolved(InternedString name);
  Type getArray(Type elemType, std::size_t size);
  Type getPointer(Type pointee);
  Type getFunction(std::span<const Type> params, std::span<const Type> returns);
  Type getDef(const Def *def);

private:
  /// Structural hash of a \c TypeImpl, for the dedup set.
  struct TypeImplHash {
    std::size_t operator()(const impl::TypeImpl *t) const { return t->hash(); }
  };

  /// Structural equality of two \c TypeImpls, for the dedup set.
  struct TypeImplEq {
    bool operator()(const impl::TypeImpl *a, const impl::TypeImpl *b) const {
      return *a == *b;
    }
  };

  /// Return the canonical \c Type for the stack-constructed \p candidate,
  /// copying it into the pool only if no structurally equal type already
  /// exists. \p size and \p align describe the candidate's storage.
  Type dedupe(std::size_t size, std::size_t align, impl::TypeImpl *candidate);

  BumpAllocator _alloc;
  std::unordered_set<impl::TypeImpl *, TypeImplHash, TypeImplEq> _types;
};

} // namespace dblang::ir
