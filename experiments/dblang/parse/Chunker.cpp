#include "parse/Chunker.h"
#include "parse/Location.h"
#include <cassert>
#include <string_view>
#include <vector>

namespace dblang {

void chunk(std::string_view filename, std::string_view source,
           std::vector<Chunk> &chunks) {
  auto addChunk = [&](std::size_t startOffset, std::size_t endOffset,
                      InFilePos startPos, InFilePos endPos) {
    auto text = source.substr(startOffset, endOffset - startOffset);
    chunks.push_back(Chunk{Loc{filename, startPos, endPos}, text});
  };

  // Offset and start of the current chunk
  std::size_t chunkStartOffset = 0;
  auto chunkStartPos = InFilePos::startOfFile();

  // Where we are currently scanning in the file
  std::size_t offset = 0;
  auto pos = InFilePos::startOfFile();

  if (source.starts_with("def ")) {
    // No header present, but we should still produce a header chunk.
    addChunk(0, 0, pos, pos);
  }

  while (offset < source.size()) {
    auto cur = source[offset];
    if (cur != '\n') {
      offset++;
      pos.column++;
      continue;
    }

    // Newline
    offset++;
    pos.line++;
    pos.column = 1;

    // Does this line begin a new definition?
    if (source.substr(offset, 4) == "def ") {
      // end previous def (if any).
      if (chunkStartOffset < offset) {
        addChunk(chunkStartOffset, offset, chunkStartPos, pos);
      }

      // Record the start of the next definitions
      chunkStartOffset = offset;
      chunkStartPos = pos;
    }
  }

  if (chunkStartOffset < offset) {
    // End the final chunk
    addChunk(chunkStartOffset, offset, chunkStartPos, pos);
  }
}

} // namespace dblang