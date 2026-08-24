#include "parse/Chunker.h"
#include "parse/Location.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <cassert>
#include <optional>
#include <string_view>
#include <vector>

namespace dblang {

namespace {

class Chunker {
private:
  std::string_view filename;
  std::string_view buffer;
  std::size_t offset = 0;
  InFilePos pos;

  std::optional<char> cur() {
    if (offset < buffer.size()) {
      return buffer[offset];
    } else {
      return std::nullopt;
    }
  }

  std::optional<std::string_view> peek(std::size_t n) {
    if (offset + n < buffer.size()) {
      return buffer.substr(offset, n);
    } else {
      return std::nullopt;
    }
  }

  void eat() {
    if (cur() == '\n') {
      pos.column = 1;
      pos.line += 1;
    } else {
      pos.column += 1;
    }

    offset++;
  }

  Loc locAt(InFilePos pos) { return Loc{filename, pos}; }

  LogicalResult chunkDef(std::vector<Chunk> &chunks);

public:
  Chunker(std::string_view filename, std::string_view source)
      : filename(filename), buffer(source), pos(InFilePos::startOfFile()) {}

  LogicalResult chunk(std::vector<Chunk> &chunks);
};

} // namespace

LogicalResult Chunker::chunkDef(std::vector<Chunk> &chunks) {
  assert(peek(4) == "def ");
  auto start = pos;
  std::size_t startOffset = offset;
  eat();
  eat();
  eat();

  while (cur() && cur() != '{') {
    eat();
  }

  if (!cur()) {
    return reportError(locAt(pos), "expected '{' after 'def' keyword");
  }

  assert(cur() == '{');
  eat();

  // Eat until the matching closing '}' (keep track of current depth).
  std::size_t depth = 1;
  while (cur() && depth > 0) {
    if (cur() == '{') {
      depth++;
    } else if (cur() == '}') {
      depth--;
    }
    eat();
  }

  if (depth > 0) {
    return reportError(locAt(pos), "expected '}' to close 'def' block");
  }

  auto text = buffer.substr(startOffset, offset - startOffset);
  chunks.push_back(Chunk{Loc{filename, start, pos}, text});
  return LogicalResult::success();
}

LogicalResult Chunker::chunk(std::vector<Chunk> &chunks) {
  // If the file starts with a def immediately.
  if (peek(4) == "def ") {
    if (failed(chunkDef(chunks))) {
      return LogicalResult::failure();
    }
  }

  while (cur()) {
    if (peek(5) == "\ndef ") {
      eat();
      if (failed(chunkDef(chunks))) {
        return LogicalResult::failure();
      }
    } else {
      eat();
    }
  }

  return LogicalResult::success();
}

LogicalResult chunk(std::string_view filename, std::string_view source,
                    std::vector<Chunk> &chunks) {
  Chunker chunker(filename, source);
  return chunker.chunk(chunks);
}

} // namespace dblang