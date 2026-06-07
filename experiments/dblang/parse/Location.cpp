#include "parse/Location.h"
#include <ostream>

namespace dblang {

std::ostream &operator<<(std::ostream &os, const Loc &loc) {
  os << loc.filename;
  if (loc.start.line) {
    os << ":" << loc.start.line;
    if (loc.start.column) {
      os << ":" << loc.start.column;
    }

    // Only print loc end if we had a start.
    if (loc.end.line) {
      if (loc.end.line == loc.start.line) {
        // Only have to print the column extent
        os << "-";
        os << loc.end.column;
      } else {
        os << "-";
        os << loc.end.line;
        if (loc.end.column) {
          os << ":" << loc.end.column;
        }
      }
    }
  }

  return os;
}

} // namespace dblang