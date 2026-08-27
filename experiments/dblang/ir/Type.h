#pragma once

#include <cstddef>
#include <functional>

namespace dblang::ir {

namespace impl {
struct TypeImpl;
}

/** Handle to type. */
class Type {
  friend class TypePool;

private:
  const impl::TypeImpl *_impl;
  /// Wrap an interned \c TypeImpl. Types are deduplicated by the \c TypePool,
  /// so two \c Type values compare equal iff they wrap the same impl.
  explicit Type(const impl::TypeImpl *impl) : _impl(impl) {}

public: /// Default-construct an empty (null) type.
  Type() : _impl(nullptr) {}

  // overload == to check that the pointers are the same
  bool operator==(const Type &other) const { return _impl == other._impl; }
  bool operator!=(const Type &other) const { return !(*this == other); }

  /// Access the underlying implementation for structural comparison.
  const impl::TypeImpl *impl() const { return _impl; }

  /// Hash of the underlying pointer. Safe because TypeImpls are deduplicated,
  /// so equal types share the same impl and thus the same hash.
  std::size_t hash() const {
    return std::hash<const impl::TypeImpl *>{}(_impl);
  }
};

} // namespace dblang::ir