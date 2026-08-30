#pragma once

#include "parse/Location.h"
#include <string_view>

namespace dblang::ir {

// integer (incl bool, char)
// string
// list
enum class LiteralKind {
  BOOL,
  CHAR,
  INT,
  STRING,
};

struct Literal {
  Loc loc;
  LiteralKind kind;
  std::string_view body;
};

} // namespace dblang::ir