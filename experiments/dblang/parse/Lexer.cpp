#include "parse/Lexer.h"
#include "parse/Location.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <optional>
#include <string_view>
#include <vector>

namespace dblang {

namespace {

class Lexer {
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

  Loc locFromTo(InFilePos start, InFilePos end) {
    return Loc{filename, start, end};
  }

  Token nextToken();

public:
  Lexer(std::string_view filename, std::string_view source)
      : filename(filename), buffer(source) {}
  LogicalResult lex(std::vector<Token> &tokens);
};

} // namespace

Token Lexer::nextToken() {
  auto token = cur();
  if (!token) {
    return Token{Token::END_OF_FILE};
  }

  auto loc = locFromTo(pos, InFilePos{pos.line, pos.column});
  reportError(loc, "unrecognized input: ") << *token;
  return Token{Token::INVALID};
}

LogicalResult Lexer::lex(std::vector<Token> &tokens) {
  while (true) {
    auto token = nextToken();
    tokens.push_back(token);
    if (token.type == Token::END_OF_FILE) {
      return LogicalResult::success();
    } else if (token.type == Token::INVALID) {
      return LogicalResult::failure();
    }
  }
}

LogicalResult lex(std::string_view filename, std::string_view source,
                  std::vector<Token> &tokens) {
  Lexer lexer(filename, source);
  return lexer.lex(tokens);
}

} // namespace dblang