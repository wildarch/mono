#pragma once

#include <string_view>

namespace dblang {

struct InFilePos {
  std::size_t line = 0;
  std::size_t column = 0;

  static InFilePos startOfFile() { return {1, 1}; }

  static InFilePos unknown() { return {0, 0}; }
};

struct Loc {
  std::string_view filename;
  InFilePos start;
  InFilePos end;
};

} // namespace dblang