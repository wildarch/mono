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

struct State {
  bool parsingInt = false;
  std::uint16_t intValue = 0;
  bool negInt = false;
  bool parsingOp = false;

  std::vector<std::int64_t> output;
  std::vector<char> ops;

  void stopIntParse() {
    parsingInt = false;
    output.push_back(negInt ? -intValue : intValue);
    intValue = 0;
    negInt = false;
    parsingOp = true;
  }

  void apply(char op) {
    auto rhs = output.back();
    output.pop_back();
    auto lhs = output.back();
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
  }

  void stepSpace() {
    if (parsingInt)
      stopIntParse();
  }

  void stepDigit(char c) {
    parsingInt = true;
    intValue *= 10;
    intValue += c - '0';
  }

  void stepAdd() {
    while (!ops.empty() && ops.back() != '(') {
      // Any op has at least the precedence of '+'.
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('+');
    parsingOp = false;
  }

  void stepSub() {
    if (!parsingOp) {
      // leading - for an int
      negInt = true;
      return;
    }

    while (!ops.empty() && ops.back() != '(') {
      // Any op has at least the precedence of '-'.
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('-');
    parsingOp = false;
  }

  void stepMul() {
    while (!ops.empty() && (ops.back() == '*' || ops.back() == '/')) {
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('*');
    parsingOp = false;
  }

  void stepDiv() {
    while (!ops.empty() && (ops.back() == '*' || ops.back() == '/')) {
      apply(ops.back());
      ops.pop_back();
    }

    ops.push_back('/');
    parsingOp = false;
  }

  void stepOpen() { ops.push_back('('); }

  void stepClose() {
    if (parsingInt)
      stopIntParse();

    while (ops.back() != '(') {
      apply(ops.back());
      ops.pop_back();
    }

    // Discard '('
    ops.pop_back();
  }

  void stepEnd() {
    if (parsingInt)
      stopIntParse();

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
      s.step(c);
    }

    s.stepEnd();

    std::cout << s.output.back() << "\n";
  }

  return 0;
}
