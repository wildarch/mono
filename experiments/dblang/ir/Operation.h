#pragma once

#include "parse/Location.h"
namespace dblang::ir {

enum class OpKind {
  // Control Flow
  IF,
  LOOP,
  RETURN,
  BREAK,
  // Expression
  LITERAL,
  CALL,
  FIELD_PTR,
  ALLOCA,
  CAST,
  LOAD,
  STORE,
  ADDRESS_OF_FUNC,
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

struct Operation {
  OpKind kind;
  Loc loc;
  Operation *prev;
  Operation *next;
  unsigned numOperands;
  unsigned numResults;
  unsigned numRegions;

  Operation *create(std::size_t numOperands, std::size_t numResults,
                    std::size_t numRegions);
};

} // namespace dblang::ir