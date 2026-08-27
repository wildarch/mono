#include "ir/StringPool.h"

namespace dblang::ir {

StringPool::StringPool() = default;

InternedString StringPool::intern(std::string_view s) {
  // Insert a copy of the string if it is not already present. The returned
  // iterator references the unique stored copy, whose data pointer is stable
  // for the lifetime of the pool.
  auto [it, _] = _strings.emplace(s);
  const auto &interned = *it;
  return InternedString(&interned);
}

} // namespace dblang::ir