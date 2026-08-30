#pragma once

namespace dblang::ir {

enum class OpKind {
  // Control Flow
  IF,
  LOOP,
  SWITCH,
  RETURN,
  BREAK,
  // Expression
  CONST,
  CALL,
  FIELD_PTR,
  ALLOCA,
  CAST,
  LOAD,
  STORE,
  // Arithmetic
  ADD,
  SUB,
  MUL,
  DIV,
  CMP,
  AND,
  OR,
  NOT,
  XOR,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  MODULO,
};

} // namespace dblang::ir