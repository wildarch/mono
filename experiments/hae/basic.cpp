/**
 * A simple and straightforward implementation.
 */
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

enum class TokenKind : std::uint8_t {
  INT,
  ADD,
  SUB,
  MUL,
  DIV,
  LPAREN,
  RPAREN,
  END_OF_LINE,
};

struct Token {
  TokenKind kind;
  std::uint16_t value;

  inline std::size_t precedence() const {
    switch (kind) {
    case TokenKind::ADD:
    case TokenKind::SUB:
      return 1;
    case TokenKind::MUL:
    case TokenKind::DIV:
      return 2;
    default:
      return 0;
    }
  }

  inline bool isBinOp() const {
    switch (kind) {
    case TokenKind::ADD:
    case TokenKind::SUB:
    case TokenKind::MUL:
    case TokenKind::DIV:
      return true;
    default:
      return false;
    }
  }
};

std::ostream &operator<<(std::ostream &os, Token const &t) {
  switch (t.kind) {
  case TokenKind::INT:
    return os << t.value;
  case TokenKind::ADD:
    return os << "+";
  case TokenKind::SUB:
    return os << "-";
  case TokenKind::MUL:
    return os << "*";
  case TokenKind::DIV:
    return os << "/";
  case TokenKind::LPAREN:
    return os << "(";
  case TokenKind::RPAREN:
    return os << ")";
  case TokenKind::END_OF_LINE:
    return os << "EOL\n";
  }
}

class Lex {
private:
  char *_buffer;
  off_t _bufferSize;
  off_t _offset = 0;
  std::vector<Token> _tokens;

  // True if we lexed a new token
  bool nextToken();
  void nextInt();

public:
  inline Lex(char *buffer, off_t bufferSize)
      : _buffer(buffer), _bufferSize(bufferSize) {}

  // True iff we lexed a new line
  bool lex();

  const auto &tokens() const { return _tokens; }
};

bool Lex::nextToken() {
  if (_offset == _bufferSize) {
    return false;
  }

  switch (_buffer[_offset]) {
  case ' ':
    // Skip
    break;
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    nextInt();
    return true;
  case '+':
    _tokens.emplace_back(Token{
        .kind = TokenKind::ADD,
    });
    break;
  case '-':
    _tokens.emplace_back(Token{
        .kind = TokenKind::SUB,
    });
    break;
  case '*':
    _tokens.emplace_back(Token{
        .kind = TokenKind::MUL,
    });
    break;
  case '/':
    _tokens.emplace_back(Token{
        .kind = TokenKind::DIV,
    });
    break;
  case '(':
    _tokens.emplace_back(Token{
        .kind = TokenKind::LPAREN,
    });
    break;
  case ')':
    _tokens.emplace_back(Token{
        .kind = TokenKind::RPAREN,
    });
    break;
  case '\n':
    _tokens.emplace_back(Token{
        .kind = TokenKind::END_OF_LINE,
    });
    break;
  default:
    std::cerr << "Invalid token: " << _buffer[_offset] << "\n";
    exit(1);
  }

  _offset++;
  return true;
}

static bool isDigit(char c) { return c >= '0' && c <= '9'; }

void Lex::nextInt() {
  std::uint16_t value = 0;
  while (_offset < _bufferSize && isDigit(_buffer[_offset])) {
    value = value * 10 + (_buffer[_offset] - '0');
    _offset++;
  }

  _tokens.emplace_back(Token{
      .kind = TokenKind::INT,
      .value = value,
  });
}

bool Lex::lex() {
  _tokens.clear();

  while (nextToken()) {
    // Lexed a new token.
    if (_tokens.back().kind == TokenKind::END_OF_LINE) {
      return true;
    }
  }

  return !_tokens.empty();
}

class Parser {
private:
  const std::vector<Token> &_tokens;
  std::size_t _offset = 0;

  inline bool done() { return _tokens.size() == _offset; }
  inline Token peek() { return _tokens[_offset]; }
  inline Token take() { return _tokens[_offset++]; }

public:
  inline Parser(const std::vector<Token> &tokens) : _tokens(tokens) {}
  std::int64_t computeAtom();
  std::int64_t computeExpr(std::size_t minPrec);
};

std::int64_t Parser::computeAtom() {
  auto token = take();
  switch (token.kind) {
  case TokenKind::INT:
    return token.value;
  case TokenKind::SUB: {
    return -computeAtom();
  }
  case TokenKind::LPAREN: {
    auto value = computeExpr(0);
    auto close = take();
    if (close.kind != TokenKind::RPAREN) {
      std::cerr << "Expected ')': " << token << "\n";
      exit(1);
    }

    return value;
  }
  default:
    std::cerr << "Not the start of an atom: " << token << "\n";
    exit(1);
  }
}

std::int64_t Parser::computeExpr(std::size_t minPrec) {
  if (done()) {
    std::cerr << "No more tokens\n";
    exit(1);
  }

  auto result = computeAtom();

  while (!done() && peek().isBinOp() && peek().precedence() >= minPrec) {
    auto binop = take();

    auto prec = binop.precedence();
    // Note: if we have MUL/DIV, there will not be a higher prec.
    auto rhs = computeExpr(prec + 1);

    // Evaluate the current token
    switch (binop.kind) {
    case TokenKind::ADD:
      result = result + rhs;
      break;
    case TokenKind::SUB:
      result = result - rhs;
      break;
    case TokenKind::MUL:
      result = result * rhs;
      break;
    case TokenKind::DIV:
      result = result / rhs;
      break;
    default:
      std::cerr << "Unexpected token " << binop << "\n";
      exit(1);
    }
  }

  return result;
}

int main(int argc, char **argv) {
  int fd = STDIN_FILENO;
  if (argc == 2) {
    fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
      perror("Cannot open input file");
      return 1;
    }
  }

  auto fsize = lseek(fd, 0, SEEK_END);
  char *buffer =
      (char *)mmap(0, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

  Lex lex(buffer, fsize);
  while (lex.lex()) {
    Parser parser(lex.tokens());
    std::cout << parser.computeExpr(0) << "\n";
  }

  return 0;
}
