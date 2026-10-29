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

struct IfOp : public Operation {
  Value cond;
  Block ifBranch;
  Block elseBranch;
};

struct LoopOp : public Operation {
  Block body;
};

struct ReturnOp : public Operation {
  Value val;
};

struct ContinueOp : public Operation {};
struct BreakOp : public Operation {};

struct ConstOp : public Operation {
  std::byte data[];
};

struct CallOp : public Operation {
  DefFunction *target;
  Value args[];
};

struct CallIndirectOp : public Operation {
  Value targetAddr;
  Value args[];
};

struct FieldPtrOp : public Operation {
  Value base;
  unsigned fieldIdx;
};

struct AllocaOp : public Operation {
  Value size;
};

} // namespace dblang::ir