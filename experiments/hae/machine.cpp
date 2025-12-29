#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <iterator>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// =================
// === DEBUGGING ===
// =================
struct dummy_out {};
template <typename T> dummy_out operator<<(dummy_out os, T s) { return os; }

#define DEBUG std::cerr
// #define DEBUG (dummy_out{})

struct Word {
  std::uint32_t val;

  void load(const char *c) { std::memcpy(&val, c, sizeof(val)); }

  char charAt(int i) {
    char chars[4];
    std::memcpy(chars, &val, 4);
    return chars[i];
  }
};

struct State {
  enum Mode {
    ATOM,
    INT,
    OP,
  } mode = ATOM;

  std::uint16_t intValue = 0;
  bool negInt = false;

  std::vector<std::int64_t> output;
  std::vector<char> ops;

  void stopIntParse() {
    assert(mode == INT);
    output.push_back(negInt ? -intValue : intValue);
    DEBUG << "output PUSH: " << output.back() << "\n";
    intValue = 0;
    negInt = false;
  }

  void apply(char op) {
    auto rhs = output.back();
    DEBUG << "output POP: " << output.back() << "\n";
    output.pop_back();
    auto lhs = output.back();
    DEBUG << "output POP: " << output.back() << "\n";
    auto &out = output.back();

    switch (op) {
    case '+':
      out = lhs + rhs;
      break;
    case '-':
      out = lhs - rhs;
      break;
    case '*':
      out = lhs * rhs;
      break;
    case '/':
      out = lhs / rhs;
      break;
    default:
      std::cerr << "invalid op '" << op << "'\n";
      std::abort();
    }

    DEBUG << "output PUSH: " << output.back() << "\n";
  }

  void stepSpace() {
    switch (mode) {
    case ATOM:
      mode = OP;
      break;
    case INT:
      stopIntParse();
      mode = OP;
      break;
    case OP:
      mode = ATOM;
      break;
    }
  }

  void stepDigit(char c) {
    if (mode == ATOM) {
      mode = INT;
    }

    assert(mode == INT);
    intValue *= 10;
    intValue += c - '0';
  }

  void stepAdd() {
    assert(mode == OP);

    while (!ops.empty() && ops.back() != '(') {
      // Any op has at least the precedence of '+'.
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('+');
    // NOTE: mode remains OP
  }

  void stepSub() {
    if (mode == ATOM) {
      // leading - for an int
      mode = INT;
      negInt = true;
      return;
    }

    assert(mode == OP);
    while (!ops.empty() && ops.back() != '(') {
      // Any op has at least the precedence of '-'.
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('-');
    // NOTE: mode remains OP
  }

  void stepMul() {
    assert(mode == OP);
    while (!ops.empty() && (ops.back() == '*' || ops.back() == '/')) {
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('*');
    // NOTE: mode remains OP
  }

  void stepDiv() {
    assert(mode == OP);
    while (!ops.empty() && (ops.back() == '*' || ops.back() == '/')) {
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('/');
    // NOTE: mode remains OP
  }

  void stepOpen() { ops.push_back('('); }

  void stepClose() {
    if (mode == INT) {
      stopIntParse();
      mode = ATOM;
    }

    while (ops.back() != '(') {
      apply(ops.back());
      ops.pop_back();
    }

    // Discard '('
    ops.pop_back();
  }

  void stepEnd() {
    if (mode == INT) {
      stopIntParse();
    }

    // Reset to initial conditions
    mode = ATOM;

    while (!ops.empty()) {
      apply(ops.back());
      ops.pop_back();
    }

    if (output.empty()) {
      DEBUG << "Nothing to output\n";
    } else if (output.size() == 1) {
      std::cout << output.back() << "\n";
      output.pop_back();
    } else {
      DEBUG << "Too much output\n";
      __builtin_unreachable();
    }
  }

  void stepChar(char c) {
    switch (c) {
    case ' ':
      return stepSpace();
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
      return stepDigit(c);
    case '+':
      return stepAdd();
    case '-':
      return stepSub();
    case '*':
      return stepMul();
    case '/':
      return stepDiv();
    case '(':
      return stepOpen();
    case ')':
      return stepClose();
    case '\n':
      return stepEnd();
    default:
      std::cerr << "Invalid token: " << c << "\n";
      exit(1);
    }
  }

  void stepWord(Word norm, Word orig) {
    for (int i = 0; i < 4; i++) {
      stepChar(orig.charAt(i));
    }
  }
};

static char normalize(char c) {
  switch (c) {
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
    // Map all integers to '0'
    return '0';
  case '+':
  case '-':
  case '*':
  case '/':
  case '(':
  case ')':
  case '\n':
  case ' ':
    return c;
  default:
    DEBUG << "Invalid token: " << c << "\n";
    __builtin_unreachable();
  }
}

// ============
// === MAIN ===
// ============
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
  char *end = buffer + fsize;

  State state;
  while (buffer < end) {
    // TODO: Optimize loads.
    char chunk[4] = {'\n', '\n', '\n', '\n'};
    auto nchars = std::min(std::distance(buffer, end), 4l);
    std::memcpy(chunk, buffer, nchars);
    buffer += nchars;

    Word orig;
    orig.load(chunk);

    char normChunk[4];
    for (int i = 0; i < 4; i++) {
      normChunk[i] = normalize(chunk[i]);
    }

    Word norm;
    norm.load(normChunk);

    state.stepWord(norm, orig);

    DEBUG << "================================\n\n";
  }

  return 0;
}
