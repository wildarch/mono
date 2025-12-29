/**
 * Implementation based on a state machine that moves one byte of input at a
 * time.
 */
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// =================
// === DEBUGGING ===
// =================
struct dummy_out {};
template <typename T> dummy_out operator<<(dummy_out os, T s) { return os; }

// #define DEBUG std::cerr
#define DEBUG (dummy_out{})

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
    if (mode == INT)
      stopIntParse();

    // Reset to initial conditions
    mode = ATOM;

    while (!ops.empty()) {
      apply(ops.back());
      ops.pop_back();
    }

    assert(output.size() == 1);
  }

  void step(char c) {
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
};

int main(int argc, char **argv) {
  std::string line;
  while (std::getline(std::cin, line)) {
    State s;
    for (auto c : line) {
      DEBUG << "input '" << c << "' mode ";
      switch (s.mode) {
      case State::ATOM:
        DEBUG << "ATOM";
        break;
      case State::INT:
        DEBUG << "INT";
        break;
      case State::OP:
        DEBUG << "OP";
        break;
      }

      DEBUG << "\n";

      s.step(c);
    }

    s.stepEnd();

    std::cout << s.output.back() << "\n";
  }

  return 0;
}
