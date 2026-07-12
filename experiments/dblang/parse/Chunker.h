#pragma once

#include "parse/Location.h"
#include "util/Result.h"
#include <string_view>
#include <vector>

namespace dblang {

struct Chunk {
  Loc loc;
  std::string_view text;
};

LogicalResult chunk(std::string_view filename, std::string_view source,
                    std::vector<Chunk> &chunks);

} // namespace dblang