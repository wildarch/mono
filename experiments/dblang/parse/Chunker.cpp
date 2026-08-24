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
  std::size_t chunkStartOffset;
  auto chunkStartPos = InFilePos::startOfFile();
  // Check if we start immediately with a def
  if (source.starts_with("def ")) {
    chunkStartOffset = 0;
  } else {
    // Reset when we see the first actual def
    chunkStartOffset = source.size();
  }

  // Where we are currently scanning in the file
  std::size_t offset = 0;
  auto pos = InFilePos::startOfFile();
  while (offset < source.size()) {
    auto cur = source[offset];
    if (cur != '\n') {
      offset++;
      pos.column++;
      continue;
    }

    // Record position before the going to the next line in case the next line
    // starts a new def
    auto chunkEndOffset = offset;
    auto chunkEndPos = pos;

    // Newline
    offset++;
    pos.line++;
    pos.column = 1;

    // Does this line begin a new definition?
    if (source.substr(offset, 4) == "def ") {
      // end previous def (if any).
      if (chunkStartOffset < chunkEndOffset) {
        addChunk(chunkStartOffset, chunkEndOffset, chunkStartPos, chunkEndPos);
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