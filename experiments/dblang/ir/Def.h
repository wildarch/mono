#pragma once

#include "ir/StringPool.h"

namespace dblang::ir {

enum class DefKind {
  STRUCT,
  ENUM,
  ALIAS, /* type alias */
  FUNC,
  CONST,  /* constant value */
  GLOBAL, /* global variable */
};

struct Def {
  InternedString name;
};

} // namespace dblang::ir