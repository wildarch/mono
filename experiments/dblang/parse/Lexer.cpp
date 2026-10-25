#include "parse/Lexer.h"
#include "parse/Location.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <cassert>
#include <filesystem>
#include <optional>
#include <string>
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
    if (offset + n <= buffer.size()) {
      return buffer.substr(offset, n);
    } else {
      return std::nullopt;
    }
  }

  /// The character n positions ahead of the current one.
  std::optional<char> lookahead(std::size_t n) {
    if (offset + n < buffer.size()) {
      return buffer[offset + n];
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

  LogicalResult eatWhitespace();

  Token nextToken();
  Token lexIdent();
  Token lexNumber();
  bool eatDigitSeq(bool (*isDigitFn)(char));
  LogicalResult eatEscapeSeq();
  Token lexChar();
  Token lexString();

public:
  Lexer(std::string_view source, Loc loc)
      : filename(loc.filename), buffer(source), pos(loc.start) {
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

LogicalResult Lexer::eatWhitespace() {
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
      auto locStart = pos;
      eat();
      eat();
      while (true) {
        if (!cur()) {
          return reportError(locFromTo(locStart, pos),
                             "unterminated block comment");
        } else if (peek(2) == "*/") {
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

  return LogicalResult::success();
}

static bool isAlpha(char c) {
  return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z');
}
static bool isDigit(char c) { return '0' <= c && c <= '9'; }
static bool isHex(char c) {
  return isDigit(c) || ('a' <= c && c <= 'f') || ('A' <= c && c <= 'F');
}
static bool isBinary(char c) { return c == '0' || c == '1'; }
static bool isAscii(char c) { return static_cast<unsigned char>(c) < 0x80; }

static bool isIdentifierStart(char c) { return isAlpha(c) || c == '_'; }
static bool isIdentifierContinue(char c) {
  return isIdentifierStart(c) || isDigit(c);
}

/// Prints a character for use in an error message.
static std::string describeChar(char c) {
  auto u = static_cast<unsigned char>(c);
  if (0x20 <= u && u < 0x7F) {
    return std::string("'") + c + "'";
  }

  const char *hex = "0123456789ABCDEF";
  return std::string("byte 0x") + hex[u >> 4] + hex[u & 0xF];
}

Token Lexer::nextToken() {
  if (failed(eatWhitespace())) {
    return Token{Loc{filename, pos, pos}, Token::INVALID};
  }

  auto locStart = pos;
  if (!cur()) {
    return Token{Loc{filename, locStart}, Token::END_OF_FILE};
  } else if (isIdentifierStart(*cur())) {
    return lexIdent();
  } else if (isDigit(*cur()) ||
             (cur() == '.' && lookahead(1) && isDigit(*lookahead(1)))) {
    return lexNumber();
  } else if (cur() == '\'') {
    return lexChar();
  } else if (cur() == '"') {
    return lexString();
  }

  // Punctuators are matched longest first.
  auto three = peek(3);
  Loc locThree{filename, pos, InFilePos{pos.line, pos.column + 3}};
#define THREE(c, t)                                                            \
  if (three == c) {                                                            \
    eat();                                                                     \
    eat();                                                                     \
    eat();                                                                     \
    return Token{locThree, Token::t, *three};                                  \
  }

  THREE("<<=", LSHIFT_ASSIGN)
  THREE(">>=", RSHIFT_ASSIGN)
#undef THREE

  auto two = peek(2);
  Loc locTwo{filename, pos, InFilePos{pos.line, pos.column + 2}};
#define TWO(c, t)                                                              \
  if (two == c) {                                                              \
    eat();                                                                     \
    eat();                                                                     \
    return Token{locTwo, Token::t, *two};                                      \
  }

  TWO("::", DCOLON)
  TWO("->", ARROW)
  TWO("++", DPLUS)
  TWO("--", DMINUS)
  TWO("&&", DAMPERSAND)
  TWO("||", DPIPE)
  TWO("==", EQUAL)
  TWO("!=", NOT_EQUAL)
  TWO("<=", LEQ)
  TWO(">=", GEQ)
  TWO("<<", LSHIFT)
  TWO(">>", RSHIFT)
  TWO("*=", ASTERISK_ASSIGN)
  TWO("/=", SLASH_ASSIGN)
  TWO("%=", PERCENT_ASSIGN)
  TWO("+=", PLUS_ASSIGN)
  TWO("-=", MINUS_ASSIGN)
  TWO("&=", AMPERSAND_ASSIGN)
  TWO("^=", CARET_ASSIGN)
  TWO("|=", PIPE_ASSIGN)
#undef TWO

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
  ONE('%', PERCENT)
  ONE('=', ASSIGN)
  ONE('!', EXCLAMATION)
  ONE('~', TILDE)
  ONE('&', AMPERSAND)
  ONE('|', PIPE)
  ONE('^', CARET)
  ONE('?', QMARK)
#undef ONE

  reportError(locOne, "unrecognized input: ") << describeChar(*cur());
  return Token{locOne, Token::INVALID};
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

/// Eats a sequence of digits, optionally separated by single underscores.
/// Returns false if there is no digit at the current position.
bool Lexer::eatDigitSeq(bool (*isDigitFn)(char)) {
  if (!cur() || !isDigitFn(*cur())) {
    return false;
  }

  eat();
  while (cur()) {
    if (isDigitFn(*cur())) {
      eat();
    } else if (cur() == '_' && lookahead(1) && isDigitFn(*lookahead(1))) {
      eat(); // the '_'
      eat();
    } else {
      break;
    }
  }

  return true;
}

Token Lexer::lexNumber() {
  auto locStart = pos;
  std::size_t start = offset;
  auto invalid = [&](std::string_view msg) {
    auto loc = locFromTo(locStart, pos);
    reportError(loc, msg);
    return Token{loc, Token::INVALID};
  };

  auto kind = Token::INT;
  auto prefix = peek(2);
  if (prefix == "0x" || prefix == "0X") {
    // hex notation, e.g. 0xDEADBEEF
    eat();
    eat();
    if (!eatDigitSeq(isHex)) {
      return invalid("expected hex digit after '0x'");
    }
  } else if (prefix == "0b" || prefix == "0B") {
    // binary notation, e.g. 0b1010
    eat();
    eat();
    if (!eatDigitSeq(isBinary)) {
      return invalid("expected binary digit after '0b'");
    }
  } else {
    bool hasIntPart = eatDigitSeq(isDigit);
    if (tryEat('.')) {
      // float notation, e.g. 3.14, 1. or .5
      kind = Token::FLOAT;
      bool hasFracPart = eatDigitSeq(isDigit);
      assert(hasIntPart || hasFracPart);
    }

    if (cur() == 'e' || cur() == 'E') {
      // exponent, e.g. 1e9
      kind = Token::FLOAT;
      eat();
      if (cur() == '+' || cur() == '-') {
        eat();
      }

      if (!eatDigitSeq(isDigit)) {
        return invalid("expected digit in exponent");
      }
    }

    if (kind == Token::INT && buffer[start] == '0' && offset - start > 1) {
      return invalid("decimal integer literal cannot start with '0'");
    }
  }

  if (cur() && isIdentifierContinue(*cur())) {
    char c = *cur();
    eat();
    reportError(locFromTo(locStart, pos), "invalid character ")
        << describeChar(c) << " in numeric literal";
    return Token{locFromTo(locStart, pos), Token::INVALID};
  }

  auto body = buffer.substr(start, offset - start);
  return Token{locFromTo(locStart, pos), kind, body};
}

LogicalResult Lexer::eatEscapeSeq() {
  assert(cur() == '\\');
  auto escStart = pos;
  eat();
  if (!cur() || cur() == '\n') {
    // Reported by the caller as unterminated.
    return LogicalResult::success();
  }

  switch (*cur()) {
  case '\'':
  case '"':
  case '?':
  case '\\':
  case 'a':
  case 'b':
  case 'f':
  case 'n':
  case 'r':
  case 't':
  case 'v':
    eat();
    return LogicalResult::success();
  case 'x':
    eat();
    if (!cur() || !isHex(*cur())) {
      return reportError(locFromTo(escStart, pos),
                         "expected hex digit in escape sequence");
    }

    while (cur() && isHex(*cur())) {
      eat();
    }

    return LogicalResult::success();
  default: {
    char c = *cur();
    eat();
    return reportError(locFromTo(escStart, pos), "invalid escape sequence: ")
           << describeChar(c);
  }
  }
}

Token Lexer::lexChar() {
  assert(*cur() == '\'');
  auto locStart = pos;
  eat();
  std::size_t start = offset;
  auto invalid = [&](std::string_view msg) {
    auto loc = locFromTo(locStart, pos);
    reportError(loc, msg);
    return Token{loc, Token::INVALID};
  };

  if (!cur() || cur() == '\n') {
    return invalid("unterminated character literal");
  } else if (cur() == '\'') {
    eat();
    return invalid("empty character literal");
  } else if (cur() == '\\') {
    if (failed(eatEscapeSeq())) {
      return Token{locFromTo(locStart, pos), Token::INVALID};
    }
  } else if (!isAscii(*cur())) {
    eat();
    return invalid("non-ASCII character in character literal");
  } else {
    eat();
  }

  if (!tryEat('\'')) {
    if (!cur() || cur() == '\n') {
      return invalid("unterminated character literal");
    }

    eat();
    return invalid("character literal must contain exactly one character");
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
    if (!cur() || cur() == '\n') {
      auto loc = locFromTo(locStart, pos);
      reportError(loc, "unterminated string literal");
      return Token{loc, Token::INVALID};
    } else if (cur() == '"') {
      eat();
      break;
    } else if (cur() == '\\') {
      if (failed(eatEscapeSeq())) {
        return Token{locFromTo(locStart, pos), Token::INVALID};
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

LogicalResult lex(std::string_view source, Loc sourceLoc,
                  std::vector<Token> &tokens) {
  Lexer lexer(source, sourceLoc);
  return lexer.lex(tokens);
}

} // namespace dblang