#include "parse/Location.h"
#include <ostream>

namespace dblang {

std::ostream &operator<<(std::ostream &os, const Loc &loc) {
  // os << loc.filename << " lines " << loc.start.line << "-" << loc.end.line <<
  // " characters " << loc.start.column << "-" << loc.end.column;
  os << loc.filename;
  if (loc.start.line) {
    // Have line info
    if (loc.end.line == loc.start.line) {
      // Single line, use short format <line>:<start.column>-<end.column>
      os << ":" << loc.start.line << ":" << loc.start.column << "-"
         << loc.end.column;
    } else {
      // long format lines <start.line>-<end.line>, characters
      // <start.column>-<end.column>.
      os << " lines " << loc.start.line << "-" << loc.end.line << " characters "
         << loc.start.column << "-" << loc.end.column;
    }
  }

  return os;
}

} // namespace dblang