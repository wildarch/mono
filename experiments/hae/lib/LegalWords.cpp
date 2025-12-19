#include "LegalWords.h"

static bool isDigit(char c) { return c >= '0' && c <= '9'; }

// Note: not considering different integers.
static constexpr std::string_view LEGAL_CHARACTERS = " ()*+-/0";

static constexpr std::string_view follows(char c) {
  switch (c) {
  case ' ':
    return "(*+-/0";
  case '(':
    return "(-0";
  case ')':
    return " )";
  case '*':
  case '+':
  case '/':
    return " ";
  case '-':
    return " 0";
  case '0':
    // TODO: 0 terminates a number
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return " )0";
  }
#undef NUM

  return "";
}

LegalWords::LegalWords(std::size_t n) {
  buildIllegalWords();

  Word tmp(n);
  enumerate(tmp, 0);
}

void LegalWords::buildIllegalWords() {
  illegalWords = {
      // Binary minus must be preceeded with a space
      "(- ",
      // Operators cannot follow each other (except -, which can be unary)
      "+ +",
      "+ *",
      "+ /",
      "- +",
      "- *",
      "- /",
      "* +",
      "* *",
      "* /",
      "/ +",
      // Operators cannot be followed by - if its binary
      "+ - ",
      "- - ",
      "* - ",
      "/ - ",
      "/ *",
      "/ /",
      // Atoms cannot directly follow each other
      "0 0",
      "0 (",
      "0 -0",
      ") 0",
      ") (",
      ") -0",

      // TODO: should we also make atoms surrounded by parens illegal?
      "(0)",
      "(-0)",
      "(00)",
      "(-00)",
      "(000)",
      "(-000)",
      "(0000)",
      "(-0000)",
      "(00000)",
      "(-00000)",
  };
}

void LegalWords::enumerate(Word &word, std::size_t i) {
  if (i == word.size()) {
    // Created the full word
    legalWords.insert(word);
    return;
  }

  std::string_view nextCharacters;
  if (i == 0) {
    // Initial words
    nextCharacters = LEGAL_CHARACTERS;
  } else {
    nextCharacters = follows(word[i - 1]);
  }

  for (auto c : nextCharacters) {
    word[i] = c;
    if (isLegal(std::string_view(word.data(), i + 1))) {
      enumerate(word, i + 1);
    }
  }
}

bool LegalWords::isLegal(std::string_view word) {
  // Max 5 consecutive digits
  int digitsRun = 0;
  for (auto c : word) {
    if (isDigit(c)) {
      digitsRun++;

      if (digitsRun > 5) {
        return false;
      }
    } else {
      digitsRun = 0;
    }
  }

  // Check the ruleset for illegal sequences
  for (int i = 0; i < word.size(); i++) {
    if (illegalWords.count(word.substr(i))) {
      return false;
    }
  }

  return true;
}
