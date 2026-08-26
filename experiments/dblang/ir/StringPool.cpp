#include "ir/StringPool.h"

namespace dblang::ir {

InternedString::InternedString(std::string_view s) : _s(s) {}

StringPool::StringPool() = default;

InternedString StringPool::intern(std::string_view s) {
  // Insert a copy of the string if it is not already present. The returned
  // iterator references the unique stored copy, whose data pointer is stable
  // for the lifetime of the pool.
  auto [it, _] = _strings.emplace(s);
  return InternedString(*it);
}

} // namespace dblang::ir