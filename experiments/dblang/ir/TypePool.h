#pragma once

#include "ir/Type.h"
#include "ir/TypeImpl.h"
#include "util/BumpAllocator.h"

#include <cstddef>
#include <span>
#include <unordered_set>

namespace dblang::ir {

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
  /// A single field of a struct type.
  using Field = impl::TypeImplStruct::Field;

  TypePool() = default;

  /// A primitive type (bool, char, iX, uX, f32, f64, isize, usize).
  Type primitive(impl::TypeKind kind);

  /// A type known only by its name in the source.
  Type unresolved(InternedString name);

  /// A fixed-size array type.
  Type array(Type elemType, std::size_t size);

  /// A pointer type.
  Type pointer(Type pointee);

  /// A function reference type.
  Type function(std::span<const Type> params, std::span<const Type> returns);

  /// A struct type with the given fields.
  Type struct_(std::span<const Field> fields);

  /// An enum type with the given alternatives.
  Type enum_(std::span<const InternedString> alts);

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
