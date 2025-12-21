#include <iostream>

#include "InputGen.h"

char InputGen::getValidOp(std::int64_t lhs, std::int64_t rhs,
                          std::size_t maxPrec) {
  char ops[4];
  std::size_t nops = 0;

  // dummy output
  std::int64_t result;

  if (!__builtin_saddl_overflow(lhs, rhs, &result)) {
    ops[nops++] = '+';
  }

  if (!__builtin_ssubl_overflow(lhs, rhs, &result)) {
    ops[nops++] = '-';
  }

  if (maxPrec >= 1 && !__builtin_smull_overflow(lhs, rhs, &result)) {
    ops[nops++] = '*';
  }

  if (maxPrec >= 1 && rhs != 0) {
    ops[nops++] = '/';
  }

  if (nops == 0) {
    std::cerr << "No valid op\n";
    exit(1);
  }

  return ops[rand() % nops];
}

std::int64_t InputGen::genAtom() {
  int opts = 3;
  if (_budget <= 1) {
    opts--;
  }

  switch (rand() % opts) {
  case 0: {
    // Positive int
    auto val = std::uint16_t(rand());
    _budget--;
    _expr += std::to_string(val);
    return val;
  }
  case 1: {
    // Negative int
    auto val = -std::int64_t(std::uint16_t(rand()));
    _budget--;
    _expr += std::to_string(val);
    return val;
  }
  case 2: {
    // Nested expr
    _expr += "(";
    auto val = genExpr();
    _expr += ")";
    return val;
  }
  }

  std::cerr << "Unreachable\n";
  exit(1);
}

std::int64_t InputGen::genExpr() {
  auto result = genAtom();
  int maxPrec = 1;
  while (_budget > 0) {
    auto lhs = result;

    _expr += " X ";
    auto opIdx = _expr.size() - 2;

    auto rhs = genAtom();

    char &op = _expr[opIdx];
    op = getValidOp(lhs, rhs, maxPrec);
    switch (op) {
    case '+':
      result = lhs + rhs;
      maxPrec = 0;
      break;
    case '-':
      result = lhs - rhs;
      maxPrec = 0;
      break;
    case '*':
      result = lhs * rhs;
      break;
    case '/':
      result = lhs / rhs;
      break;
    default:
      std::cerr << "Invalid op " << op << "\n";
    }
  }

  return result;
}

std::pair<std::string, std::int64_t> InputGen::genExpr(std::size_t budget) {
  _budget = budget;
  auto result = genExpr();
  return {
      std::move(_expr),
      result,
  };
}
