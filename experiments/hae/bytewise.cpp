/**
 * Implementation based on a state machine that moves one byte of input at a
 * time.
 */
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

struct State {
  bool parsingInt = false;
  std::uint16_t intValue = 0;
  int minPrec = 0;

  std::int64_t result = 0;

  void stopIntParse() {
    parsingInt = false;
    result = intValue;
    intValue = 0;
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
    if (minPrec > 1) {
      // At MUL/DIV level.
      // Keep result.
      std::abort();
    } else {
      // TODO: compute rhs
      // TODO: result = result + rhs;
      std::abort();
    }
  }

  void stepSub() {
    // TODO: implement
    std::abort();
  }

  void stepMul() {
    // TODO: implement
    std::abort();
  }

  void stepDiv() {
    // TODO: implement
    std::abort();
  }

  void stepOpen() {
    // TODO: Save current result.
    // TODO: Parse a new atom.
    std::abort();
  }

  void stepClose() {
    if (parsingInt)
      stopIntParse();

    // TODO: pop from continue stack
    std::abort();
  }

  void stepEnd() {
    // Nothing, result is already set.
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
  }

  return 0;
}
