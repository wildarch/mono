#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "ir/Def.h"
#include "ir/Type.h"
#include "parse/Location.h"
namespace dblang::ir {

#define GEN_OP_KIND
#include "Operation.inc.h"

struct Context {
  // type pool
  // constant pool
  // OOL operand pool
};

struct Operation {
  OpKind kind;
  Loc loc;
  Type resultType;
};

struct Value {
  unsigned off;
};

struct Block {
  unsigned off;
};

enum class Comparator {
  EQ,
  NEQ,
  LT,
  LTE,
  GT,
  GTE,
};

#define GEN_OP_CLASSES
#include "Operation.inc.h"

#define GEN_OP_BUILD_DECL
#include "Operation.inc.h"

} // namespace dblang::ir