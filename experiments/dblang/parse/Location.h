#pragma once

#include <string_view>

namespace dblang {

struct InFilePos {
  std::size_t line = 1;
  std::size_t column = 1;
};

struct Loc {
  std::string_view filename;
  InFilePos start;
  InFilePos end;
};

} // namespace dblang