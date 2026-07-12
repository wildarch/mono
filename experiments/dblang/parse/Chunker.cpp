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
  LogicalResult chunk(std::vector<Chunk> &chunks);
};

} // namespace

LogicalResult Chunker::chunkDef(std::vector<Chunk> &chunks) {
  assert(peek(4) == "\ndef");
  auto start = pos;
  eat();
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

  // TODO: eat until closing '}' (keep track of current depth)

  return LogicalResult::success();
}

LogicalResult Chunker::chunk(std::vector<Chunk> &chunks) {
  auto pos = InFilePos::startOfFile();
  std::size_t offset = 0;

  while (cur()) {
    if (peek(4) == "\ndef") {
      if (failed(chunkDef(chunks))) {
        return LogicalResult::failure();
      }
    } else {
      eat();
    }
  }

  return LogicalResult::success();
}

} // namespace dblang