#pragma once

#include <cstdint>
#include <string>

class InputGen {
private:
  std::string _expr;
  std::size_t _budget = 0;

  char getValidOp(std::int64_t lhs, std::int64_t rhs, std::size_t maxPrec);
  std::int64_t genAtom();
  std::int64_t genExpr();

public:
  std::pair<std::string, std::int64_t> genExpr(std::size_t budget);
};
