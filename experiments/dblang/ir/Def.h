#pragma once

#include "ir/StringPool.h"
#include "ir/Type.h"
#include "parse/Location.h"

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
  Loc loc;
  InternedString name;
};

struct DefStruct {
  struct Field {
    InternedString name;
    Type type;
  };
  Field fields[];

  static constexpr std::size_t computeAllocationSize(std::size_t nFields) {
    return sizeof(DefStruct) + nFields * sizeof(Field);
  }
};

struct DefEnum : public Def {
  std::size_t nAlts;
  using Alt = InternedString;
  Alt alts[];

  static constexpr std::size_t computeAllocationSize(std::size_t nFields) {
    return sizeof(DefEnum) + nFields * sizeof(Alt);
  }
};

struct DefAlias : public Def {
  Type aliased;
};

struct DefFunction : public Def {
  FunctionType type;

  // TODO body
};

struct DefConst : public Def {
  // TODO expression body
};

struct DefGlobal : public Def {
  Type type;
};

} // namespace dblang::ir