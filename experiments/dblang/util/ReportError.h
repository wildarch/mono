#pragma once

#include "parse/Location.h"
#include "util/Result.h"
#include <sstream>
#include <string_view>

namespace dblang {

class InFlightDiagnostic {
private:
  Loc loc;
  std::stringstream buffer;
  bool reported = false;

  void ensureReported();

  /// Append arguments to the diagnostic.
  template <typename Arg> InFlightDiagnostic &append(Arg &&arg) & {
    buffer << std::forward<Arg>(arg);
    return *this;
  }
  template <typename Arg> InFlightDiagnostic &&append(Arg &&arg) && {
    return std::move(append(std::forward<Arg>(arg)));
  }

public:
  InFlightDiagnostic(Loc loc, std::string_view msg) : loc(loc) {
    buffer << msg;
  }
  ~InFlightDiagnostic();

  template <typename Arg> InFlightDiagnostic &operator<<(Arg &&arg) & {
    return append(std::forward<Arg>(arg));
  }
  template <typename Arg> InFlightDiagnostic &&operator<<(Arg &&arg) && {
    return std::move(append(std::forward<Arg>(arg)));
  }

  operator LogicalResult() const;
};

InFlightDiagnostic reportError(Loc loc, std::string_view msg);

} // namespace dblang