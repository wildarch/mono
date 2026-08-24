#pragma once

#include "parse/Location.h"
#include <string_view>
#include <vector>

namespace dblang {

struct Chunk {
  Loc loc;
  std::string_view text;
};

/**
 * Split a file into individual top-level definitions.
 *
 * The first chunk (always present) contains the file header, that is everything
 * before the first definition.
 *
 * Every definition starts with the keyword "def", which must be at the start of
 * a line.
 */
void chunk(std::string_view filename, std::string_view source,
           std::vector<Chunk> &chunks);

} // namespace dblang