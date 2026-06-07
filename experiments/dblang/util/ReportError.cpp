#include "util/ReportError.h"
#include "parse/Location.h"
#include "util/Result.h"
#include <iostream>

namespace dblang {

void InFlightDiagnostic::ensureReported() {
  if (reported) {
    // Already reported
    return;
  }

  std::cerr << loc << ": error: " << buffer.str() << "\n";
  buffer.clear();
  reported = true;
}

InFlightDiagnostic::~InFlightDiagnostic() { ensureReported(); }

InFlightDiagnostic::operator LogicalResult() const {
  return LogicalResult::failure();
}

InFlightDiagnostic reportError(Loc loc, std::string_view msg) {
  return InFlightDiagnostic(loc, msg);
}

} // namespace dblang