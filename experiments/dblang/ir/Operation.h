#pragma once

#include "ir/Def.h"
#include "ir/Literal.h"
#include "ir/Type.h"
#include "parse/Location.h"
namespace dblang::ir {

enum class OpKind {
  // Control Flow
  IF,
  LOOP,
  RETURN,
  CONTINUE,
  BREAK,
  // Expression
  LITERAL,
  CALL,
  CALL_MEMBER,
  CALL_INDIRECT,
  FIELD_PTR,
  ALLOCA,
  CAST,
  LOAD,
  STORE,
  FUNC_ADDR,
  GLOBAL_ADDR,
  CONST_ADDR,
  // Arithmetic
  ADD,
  SUB,
  MUL,
  DIV,
  MODULO,
  CMP,
  BIT_AND,
  BIT_OR,
  BIT_NOT,
  BIT_XOR,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  PTR_ADD,
  PTR_SUB,
};

struct Context {
  // type pool
  // constant pool
  // OOL operand pool
};

struct OpHead {
  OpKind kind;
  Loc loc;
  Type resultType;
};

struct Operation : public OpHead {};

using Value = unsigned;
using Block = unsigned;

struct IfOp {
  Value cond;
  Block ifBranch;
  Block elseBranch;
};

struct LoopOp {
  Block body;
};

struct ReturnOp {
  Value val;
};

struct LiteralOp {
  Literal val;
};

// Same idea for member and indirect versions (for those the first arg is
// special)
struct CallOp {
  DefFunction *target;
  unsigned nargs;
  void *ool;
};

struct BinOp {
  Value lhs, rhs;
};

} // namespace dblang::ir