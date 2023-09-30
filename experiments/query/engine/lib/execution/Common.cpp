#include "execution/Common.h"
#include <ostream>
#include <string_view>

namespace execution {

std::ostream &operator<<(std::ostream &os, const SmallString &s) {
  os << std::string_view(s);
  return os;
}

} // namespace execution