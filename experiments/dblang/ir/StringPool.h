#pragma once

#include <string>
#include <string_view>
#include <unordered_set>

namespace dblang::ir {

class InternedString {
  friend class StringPool;

private:
  // is 16 bytes, maybe store a pointer instead to keep to 8 bytes?
  const std::string *_s;
  InternedString(const std::string *s) : _s(s) {}

public:
  /// Dereference to the underlying string view.
  std::string_view operator*() const { return *_s; }

  /// Cheap equivalence: interned strings with equal content share storage, so
  /// comparing the data pointers alone is sufficient.
  bool operator==(const InternedString &other) const {
    return _s->data() == other._s->data();
  }
  bool operator!=(const InternedString &other) const {
    return !(*this == other);
  }
};

/**
 * Stores unique string values.
 *
 * Provides callers with a string reference that remains valid for the lifetime
 * of the pool. Two interned strings can be compared for equivalence cheaply, as
 * two (distinct) strings with the same content correspond to the same \c
 * InternedString after a call to \c intern.
 */
class StringPool {
private:
  // Storage for the unique string contents. std::string's small-string
  // optimization means short strings live inline, so their data() pointers
  // are not stable across moves; keeping the set of std::string stable (via
  // node-based std::unordered_set) guarantees the pointers remain valid for
  // the lifetime of the pool.
  std::unordered_set<std::string> _strings;

public:
  StringPool();

  InternedString intern(std::string_view s);
};

} // namespace dblang::ir

namespace std {

template <> struct hash<dblang::ir::InternedString> {
  std::size_t operator()(const dblang::ir::InternedString &s) const {
    // Interned strings with equal content share storage, so hashing the
    // underlying string view is a valid (and cheap) content hash.
    return std::hash<std::string_view>{}(*s);
  }
};

} // namespace std