#include <array>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

// =================
// === DEBUGGING ===
// =================
struct dummy_out {};
template <typename T> dummy_out operator<<(dummy_out os, T s) { return os; }

// #define DEBUG std::cerr
#define DEBUG (dummy_out{})

#define CHECK

// ========================
// === FAST INT PARSING ===
// ========================
bool isDigit(char c) { return c >= '0' && c <= '9'; }

static std::uint16_t parseInt(char *&curr) {
  // The largest possible integer is UINT16_MAX (65535), so at most 5 digits
  std::uint8_t d0 = std::uint8_t(curr[0]) - '0';
  std::uint8_t d1 = std::uint8_t(curr[1]) - '0';
  std::uint8_t d2 = std::uint8_t(curr[2]) - '0';
  std::uint8_t d3 = std::uint8_t(curr[3]) - '0';
  std::uint8_t d4 = std::uint8_t(curr[4]) - '0';

  // Assuming that integers are randomly sampled, we expect the vast majority to
  // have 5 digits:
  // - 1 digit: 10
  // - 2 digits: 90
  // - 3 digits: 900
  // - 4 digits: 9000
  // - 5 digits: 55536

  if (d1 <= 9 && d2 <= 9 && d3 <= 9 && d4 <= 9) [[likely]] {
    // 5 digits
    curr += 5;
    return std::uint16_t(d0) * 10000 + std::uint16_t(d1) * 1000 +
           std::uint16_t(d2) * 100 + std::uint16_t(d3) * 10 + std::uint16_t(d4);
  } else if (d1 <= 9 && d2 <= 9 && d3 <= 9) {
    // 4 digits
    curr += 4;
    return std::uint16_t(d0) * 1000 + std::uint16_t(d1) * 100 +
           std::uint16_t(d2) * 10 + std::uint16_t(d3);
  } else if (d1 <= 9 && d2 <= 9) {
    // 3 digits
    curr += 3;
    return std::uint16_t(d0) * 100 + std::uint16_t(d1) * 10 + std::uint16_t(d2);
  } else if (d1 <= 9) {
    // 3 digits
    curr += 2;
    return std::uint16_t(d0) * 10 + std::uint16_t(d1);
  } else {
    curr += 1;
    return d0;
  }
}

// TODO: optimize
static std::int64_t parseSignedInt(char *&curr) {
  DEBUG << "Expect - or digit: " << *curr << "\n";
  // Note: both branches are equally likely.
  if (*curr == '-') {
    return -std::int64_t(parseInt(++curr));
  } else {
    return parseInt(curr);
  }
}

// =============
// === STATE ===
// =============
struct Continuation;

#define STATE_PARAMS                                                           \
  char *curr, char *end, Continuation *stack, std::int64_t a, std::int64_t b,  \
      std::int64_t c, char **lineEnd

#define STATE_PARAMS_REF                                                       \
  char *&curr, char *&end, Continuation *&stack, std::int64_t a,               \
      std::int64_t b, std::int64_t c, char **&lineEnd

#define STATE_ARGS curr, end, stack, a, b, c, lineEnd

// ==================================
// === CONTINUATION (DECLARATION) ===
// ==================================
using ContinuationFunction = auto(STATE_PARAMS) -> std::int64_t;

struct Continuation {
  ContinuationFunction *func;
  std::int64_t a;
  std::int64_t b;
};

static inline void push(Continuation *&stack, ContinuationFunction *func) {
  *(++stack) = Continuation{
      .func = func,
  };
}

static inline void push(Continuation *&stack, std::int64_t a,
                        ContinuationFunction *func) {
  *(++stack) = Continuation{
      .func = func,
      .a = a,
  };
}

static inline void push(Continuation *&stack, std::int64_t a, std::int64_t b,
                        ContinuationFunction *func) {
  *(++stack) = Continuation{
      .func = func,
      .a = a,
      .b = b,
  };
}

#define POP                                                                    \
  {                                                                            \
    auto &cont = *(stack--);                                                   \
    a = cont.a;                                                                \
    b = cont.b;                                                                \
    DEBUG << "restore a = " << a << "\n";                                      \
    DEBUG << "restore b = " << b << "\n";                                      \
    [[clang::musttail]] return cont.func(STATE_ARGS);                          \
  }

// ==================
// === OPERATIONS ===
// ==================
struct Add {
  static constexpr int PREC = 0;
  static constexpr char SYMB = '+';
  static inline std::int64_t apply(std::int64_t a, std::int64_t b) {
    return a + b;
  }
};

struct Sub {
  static constexpr int PREC = 0;
  static constexpr char SYMB = '-';
  static inline std::int64_t apply(std::int64_t a, std::int64_t b) {
    return a - b;
  }
};

struct Mul {
  static constexpr int PREC = 1;
  static constexpr char SYMB = '*';
  static inline std::int64_t apply(std::int64_t a, std::int64_t b) {
    return a * b;
  }
};

struct Div {
  static constexpr int PREC = 1;
  static constexpr char SYMB = '/';
  static inline std::int64_t apply(std::int64_t a, std::int64_t b) {
    return a / b;
  }
};

static std::int64_t parseStart(STATE_PARAMS);
template <typename Op> static std::int64_t parseOne(STATE_PARAMS);
template <typename Op1, typename Op2>
static std::int64_t parseTwo(STATE_PARAMS);

static std::int64_t continueStart(STATE_PARAMS);
template <typename Op> static std::int64_t continueOne(STATE_PARAMS);
template <typename Op1, typename Op2>
static std::int64_t continueTwo(STATE_PARAMS);

// Called at the start of a (sub-)expression.
static std::int64_t parseStart(STATE_PARAMS) {
  DEBUG << "start parse\n";

  // Expect an atom, so either '-', '0'-'9' or '('
#ifdef CHECK
  {
    auto c = *curr;
    if (c != '-' && !isDigit(c) && c != '(') {
      std::cerr << "Not expected in parseStart: " << c << "\n";
      abort();
    }
  }
#endif

  if (*curr == '(') {
    curr++;
    push(stack, continueStart);
    [[clang::musttail]] return parseStart(STATE_ARGS);
  }

  c = parseSignedInt(curr);
  DEBUG << "c = " << c << "\n";

  [[clang::musttail]] return continueStart(STATE_ARGS);
}

// Called after the first op has been parsed
template <typename Op> static std::int64_t parseOne(STATE_PARAMS) {
  DEBUG << "parseOne<" << Op::SYMB << ">\n";

  // Next atom, either a sub-expr in parentheses or a plain int
#ifdef CHECK
  {
    auto c = *curr;
    if (c != '-' && !isDigit(c) && c != '(') {
      std::cerr << "Not expected in parseStart: " << c << "\n";
      abort();
    }
  }
#endif

  if (*curr == '(') {
    curr++;
    push(stack, a, continueOne<Op>);
    [[clang::musttail]] return parseStart(STATE_ARGS);
  }

  // continuation expects new value to be in c
  c = parseSignedInt(curr);
  DEBUG << "c = " << c << "\n";
  [[clang::musttail]] return continueOne<Op>(STATE_ARGS);
}

template <typename Op1, typename Op2>
static std::int64_t parseTwo(STATE_PARAMS) {
  DEBUG << "parseTwo<" << Op1::SYMB << ", " << Op2::SYMB << ">\n";
  if constexpr (Op1::PREC >= Op2::PREC) {
    DEBUG << "a = a " << Op1::SYMB << " b = " << a << " " << Op1::SYMB << " "
          << b << " = " << Op1::apply(a, b) << "\n";
    a = Op1::apply(a, b);
    [[clang::musttail]] return parseOne<Op2>(STATE_ARGS);
  }

#ifdef CHECK
  {
    auto c = *curr;
    if (c != '-' && !isDigit(c) && c != '(') {
      std::cerr << "Not expected in parseStart: " << c << "\n";
      abort();
    }
  }
#endif

  // Parse the last int
  if (*curr == '(') {
    curr++;
    push(stack, a, b, continueTwo<Op1, Op2>);
    [[clang::musttail]] return parseStart(STATE_ARGS);
  }

  c = parseSignedInt(curr);
  DEBUG << "c = " << c << "\n";
  [[clang::musttail]] return continueTwo<Op1, Op2>(STATE_ARGS);
}

// Called after the first integer has been parsed
static std::int64_t continueStart(STATE_PARAMS) {
  if (*curr == '\n' || curr == end) [[unlikely]] {
    *lineEnd = curr + 1;
    DEBUG << "return c = " << c << "\n";
    return c;
  } else if (*curr == ')') {
    curr++;
    POP
  }

  DEBUG << "a = c = " << c << "\n";
  a = c;

  // Find the next operator
  auto op = curr[1];
  curr += 3;
  switch (op) {
  case '+':
    [[clang::musttail]] return parseOne<Add>(STATE_ARGS);
  case '-':
    [[clang::musttail]] return parseOne<Sub>(STATE_ARGS);
  case '*':
    [[clang::musttail]] return parseOne<Mul>(STATE_ARGS);
  case '/':
    [[clang::musttail]] return parseOne<Div>(STATE_ARGS);
  }

  DEBUG << "Unexpected op '" << op << "'\n";

#ifdef CHECK
  std::cerr << "Expected op, found '" << curr[-3] << curr[-2] << curr[-1]
            << "'\n";
  abort();
#endif

  __builtin_unreachable();
}

// Called after the second int has been parsed
template <typename Op> static std::int64_t continueOne(STATE_PARAMS) {
  b = c;

  if (curr == end || *curr == '\n') [[unlikely]] {
    // Newline or end of file -> evaluate and terminate
    *lineEnd = curr + 1;
    auto r = Op::apply(a, b);
    DEBUG << "return a " << Op::SYMB << " b = " << a << " " << Op::SYMB << " "
          << b << " = " << r << "\n";
    return r;
  } else if (*curr == ')') {
    // Closing paren -> evaluate to c and POP
    curr++;
    c = Op::apply(a, b);
    DEBUG << "c = a " << Op::SYMB << " b = " << a << " " << Op::SYMB << " " << b
          << " = " << c << "\n";
    POP
  }

  // Find the next operator
  auto op = curr[1];
  curr += 3;
  switch (op) {
  case '+':
    [[clang::musttail]] return parseTwo<Op, Add>(STATE_ARGS);
  case '-':
    [[clang::musttail]] return parseTwo<Op, Sub>(STATE_ARGS);
  case '*':
    [[clang::musttail]] return parseTwo<Op, Mul>(STATE_ARGS);
  case '/':
    [[clang::musttail]] return parseTwo<Op, Div>(STATE_ARGS);
  }

  DEBUG << "Unexpected op '" << op << "'\n";

#ifdef CHECK
  std::cerr << "Expected op, found '" << curr[-3] << curr[-2] << curr[-1]
            << "'\n";
  abort();
#endif

  __builtin_unreachable();
}

template <typename Op1, typename Op2>
static std::int64_t continueTwo(STATE_PARAMS) {
  DEBUG << "c = b " << Op2::SYMB << " c =" << b << " " << Op2::SYMB << " " << c
        << " = " << (Op2::apply(b, c)) << "\n";
  c = Op2::apply(b, c);

  [[clang::musttail]] return continueOne<Op1>(STATE_ARGS);
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

  while (buffer < end) {
    char *lineEnd;
    constexpr std::size_t STACK_SIZE = 50000;
    std::array<Continuation, STACK_SIZE> stack;
    auto v = parseStart(buffer, end, stack.data(), 0, 0, 0, &lineEnd);

    std::cout << v << std::endl;
    buffer = lineEnd;

    DEBUG << "================================\n\n";
  }

  return 0;
}
