#include "parse/Lexer.h"
#include "parse/Location.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <cassert>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dblang {

namespace {

class Lexer {
private:
  std::unordered_map<std::string_view, Token::Kind> keywords;
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

  void initKeywords();

  void eatWhitespace();

  Token nextToken();
  Token lexIdent();
  Token lexIntOrFloat();
  Token lexChar();
  Token lexString();

public:
  Lexer(std::string_view filename, std::string_view source)
      : filename(filename), buffer(source), pos(InFilePos::startOfFile()) {
    initKeywords();
  }
  LogicalResult lex(std::vector<Token> &tokens);
};

} // namespace

void Lexer::initKeywords() {
#define CASE(KIND, STR) keywords[STR] = Token::KIND;
  DBLANG_KEYWORDS(CASE)
#undef CASE
}

void Lexer::eatWhitespace() {
  while (cur()) {
    if (cur() == ' ' || cur() == '\t' || cur() == '\n' || cur() == '\r') {
      eat();
    } else if (peek(2) == "//") {
      // line comment
      while (cur() && cur() != '\n') {
        eat();
      }
    } else if (peek(2) == "/*") {
      /* multi-line comment */
      eat();
      eat();
      while (cur()) {
        if (peek(2) == "*/") {
          eat();
          eat();
          break;
        } else {
          eat();
        }
      }
    } else {
      // No (more) whitespace
      break;
    }
  }
}

static bool isAlpha(char c) {
  return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z');
}
static bool isDigit(char c) { return '0' <= c && c <= '9'; }
static bool isHex(char c) {
  return isDigit(c) || ('a' <= c && c <= 'f') || ('A' <= c && c <= 'F');
}

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

  // Ellipsis
  auto three = peek(3);
  Loc locThree{filename, pos, InFilePos{pos.line, pos.column + 3}};
  if (three == "...") {
    eat();
    eat();
    eat();
    return Token{locThree, Token::ELLIPSIS, *three};
  }

  // Operators of length 2
  auto two = peek(2);
  Loc locTwo{filename, pos, InFilePos{pos.line, pos.column + 2}};
#define TWO(c, t)                                                              \
  if (two == c) {                                                              \
    eat();                                                                     \
    eat();                                                                     \
    return Token{locTwo, Token::t, *two};                                      \
  }

  TWO("->", ARROW)
  TWO("&&", DAMPERSAND)
  TWO("||", DPIPE)
  TWO("==", EQUAL)
  TWO("!=", NOT_EQUAL)
  TWO("<=", LEQ)
  TWO(">=", GEQ)
  TWO("++", INC)
  TWO("--", DEC)
  TWO("<<", LSHIFT)
  TWO(">>", RSHIFT)
  TWO("+=", PLUS_EQ)
  TWO("-=", MINUS_EQ)
#undef TWO

  // Operators and punctuation (length 1)
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
  ONE('*', ASTERISK)
  ONE('/', SLASH)
  ONE('=', ASSIGN)
  ONE('!', EXCLAMATION)
  ONE('~', TILDE)
  ONE('&', AMPERSAND)
  ONE('|', PIPE)
  ONE('?', QMARK)
  ONE('%', PERCENT)
  ONE('^', CARET)

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

  // Check if we have a keyword instead of an identifier.
  auto it = keywords.find(body);
  auto kind = Token::IDENT;
  if (it != keywords.end()) {
    // keyword!
    kind = it->second;
  }

  return Token{locFromTo(locStart, pos), kind, body};
}

Token Lexer::lexIntOrFloat() {
  assert(isDigit(*cur()));
  auto locStart = pos;
  std::size_t start = offset;
  while (cur() && isDigit(*cur())) {
    eat();
  }

  auto body = buffer.substr(start, offset - start);
  if (body == "0" && cur() == 'x') {
    // hex notation, e.g. 0xDEADBEEF
    eat(); // the 'x'
    while (cur() && isHex(*cur())) {
      eat();
    }

    body = buffer.substr(start, offset - start);
    if (body == "0x") {
      auto loc = locFromTo(locStart, pos);
      reportError(loc,
                  "invalid hex integer (must have at least one hex digit)");
      return Token{loc, Token::INVALID};
    }

    return Token{locFromTo(locStart, pos), Token::INT, body};
  }

  // float notation, e.g. 3.14
  if (cur() == '.') {
    eat(); // the '.'
    while (cur() && isDigit(*cur())) {
      eat();
    }

    body = buffer.substr(start, offset - start);
    return Token{locFromTo(locStart, pos), Token::FLOAT, body};
  }

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
  auto body = buffer.substr(start, offset - start - 1);
  return Token{loc, Token::CHAR, body};
}

Token Lexer::lexString() {
  assert(*cur() == '"');
  auto locStart = pos;
  eat();
  std::size_t start = offset;
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

  auto body = buffer.substr(start, offset - start - 1);
  return Token{locFromTo(locStart, pos), Token::STRING, body};
}

LogicalResult Lexer::lex(std::vector<Token> &tokens) {
  while (true) {
    auto token = nextToken();
    if (token.kind == Token::END_OF_FILE) {
      return LogicalResult::success();
    } else if (token.kind == Token::INVALID) {
      return LogicalResult::failure();
    }

    tokens.push_back(token);
  }
}

std::ostream &operator<<(std::ostream &os, const Token &token) {
  return os << token.loc << ": " << Token::kindName(token.kind) << " '"
            << token.body << "'";
}

const char *Token::kindName(Kind k) {
  switch (k) {
#define CASE(X)                                                                \
  case X:                                                                      \
    return #X;

    DBLANG_ENUM_TOKEN_KIND(CASE)
#undef CASE

    // keywords
#define CASE(X, _Y)                                                            \
  case X:                                                                      \
    return #X;

    DBLANG_KEYWORDS(CASE)
#undef CASE
  }
}

LogicalResult lex(std::string_view filename, std::string_view source,
                  std::vector<Token> &tokens) {
  Lexer lexer(filename, source);
  return lexer.lex(tokens);
}

} // namespace dblang