#include "parse/Lexer.h"
#include "parse/Location.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <cassert>
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

  void eat() {
    if (cur() == '\n') {
      pos.column = 1;
      pos.line += 1;
    } else {
      pos.column += 1;
    }

    offset++;
  }

  bool tryEat(char c) {
    if (cur() == c) {
      eat();
      return true;
    } else {
      return false;
    }
  }

  Loc locFromTo(InFilePos start, InFilePos end) {
    return Loc{filename, start, end};
  }

  void eatWhitespace();

  Token nextToken();
  Token lexIdent();
  Token lexIntOrFloat();
  Token lexChar();
  Token lexString();

public:
  Lexer(std::string_view filename, std::string_view source)
      : filename(filename), buffer(source), pos(InFilePos::startOfFile()) {}
  LogicalResult lex(std::vector<Token> &tokens);
};

} // namespace

void Lexer::eatWhitespace() {
  while (cur()) {
    if (cur() == ' ') {
      eat();
    } else if (cur() == '\n') {
      eat();
    }
    // TODO: line comment
    else {
      return;
    }
  }
}

static bool isAlpha(char c) {
  return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z');
}
static bool isDigit(char c) { return '0' <= c && c <= '9'; }

static bool isIdentifierStart(char c) { return isAlpha(c) || c == '_'; }
static bool isIdentifierContinue(char c) {
  return isIdentifierStart(c) || isDigit(c);
}

Token Lexer::nextToken() {
  eatWhitespace();
  auto locStart = pos;
  if (!cur()) {
    return Token{Loc{filename, locStart}, Token::END_OF_FILE};
  } else if (isIdentifierStart(*cur())) {
    return lexIdent();
  } else if (isDigit(*cur())) {
    return lexIntOrFloat();
  } else if (cur() == '\'') {
    return lexChar();
  } else if (cur() == '"') {
    return lexString();
  }

  auto one = buffer.substr(offset, 1);
  Loc locOne{filename, locStart, InFilePos{locStart.line, locStart.column + 1}};
#define ONE(c, t)                                                              \
  if (tryEat(c)) {                                                             \
    return Token{locOne, Token::t, one};                                       \
  }

  ONE('(', LPAREN)
  ONE(')', RPAREN)
  ONE('{', LBRACE)
  ONE('}', RBRACE)
  ONE('[', LSBRACKET)
  ONE(']', RSBRACKET)
  ONE('<', LANGLE)
  ONE('>', RANGLE)
  ONE(':', COLON)
  ONE(',', COMMA)
  ONE('.', DOT)
  ONE(';', SEMI)
  ONE('+', PLUS)
  ONE('-', MINUS)
  ONE('*', TIMES)
  ONE('/', DIVIDE)
  ONE('=', ASSIGN)
  ONE('!', NOT)
  ONE('~', BITNOT)
  ONE('&', BITAND)
  ONE('|', BITOR)
  ONE('?', TERNARY)
  ONE('%', MOD)
  ONE('^', XOR)

  Loc invalidLoc{filename, locStart};
  reportError(invalidLoc, "unrecognized input: ") << *cur();
  return Token{invalidLoc, Token::INVALID};
}

Token Lexer::lexIdent() {
  assert(isIdentifierStart(*cur()));
  auto locStart = pos;
  std::size_t start = offset;
  while (cur() && isIdentifierContinue(*cur())) {
    eat();
  }

  auto body = buffer.substr(start, offset - start);
  return Token{locFromTo(locStart, pos), Token::IDENT, body};
}

Token Lexer::lexIntOrFloat() {
  // TODO: floats and other weird formats too.
  assert(isDigit(*cur()));
  auto locStart = pos;
  std::size_t start = offset;
  while (cur() && isDigit(*cur())) {
    eat();
  }

  auto body = buffer.substr(start, offset - start);
  return Token{locFromTo(locStart, pos), Token::INT, body};
}

Token Lexer::lexChar() {
  assert(*cur() == '\'');
  auto locStart = pos;
  eat();
  std::size_t start = offset;
  bool escaped;
  if (cur() == '\\') {
    // eat the escape
    escaped = true;
    eat();
  }

  if (!cur()) {
    // Unterminated
    auto loc = locFromTo(locStart, pos);
    reportError(loc, "unterminated character");
    return Token{loc, Token::INVALID};
  } else if (cur() == '\'' && !escaped) {
    // got end of character too soon
    auto loc = locFromTo(locStart, pos);
    reportError(loc, "character literal of length 0");
    return Token{loc, Token::INVALID};
  } else {
    // The character.
    eat();
  }

  if (!tryEat('\'')) {
    auto loc = locFromTo(locStart, pos);
    reportError(loc, "character literal of length > 1");
    return Token{loc, Token::INVALID};
  }

  auto loc = locFromTo(locStart, pos);
  return Token{loc, Token::CHAR};
}

Token Lexer::lexString() {
  assert(*cur() == '"');
  auto locStart = pos;
  std::size_t start = offset;
  eat();
  while (true) {
    if (!cur()) {
      auto loc = locFromTo(locStart, pos);
      reportError(loc, "unterminated string");
      return Token{loc, Token::INVALID};
    } else if (cur() == '"') {
      eat();
      break;
    } else if (cur() == '\\') {
      eat();
      // Also eat the next (escaped) character
      if (cur()) {
        eat();
      }
    } else {
      eat();
    }
  }

  auto body = buffer.substr(start, offset - start);
  return Token{locFromTo(locStart, pos), Token::STRING, body};
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

std::ostream &operator<<(std::ostream &os, const Token &token) {
  return os << token.loc << ": " << Token::kindName(token.type) << " '"
            << token.body << "'";
}

const char *Token::kindName(Kind k) {
  switch (k) {
#define CASE(X)                                                                \
  case X:                                                                      \
    return #X;

    DBLANG_ENUM_TOKEN_KIND(CASE)
#undef CASE
  }
}

LogicalResult lex(std::string_view filename, std::string_view source,
                  std::vector<Token> &tokens) {
  Lexer lexer(filename, source);
  return lexer.lex(tokens);
}

} // namespace dblang