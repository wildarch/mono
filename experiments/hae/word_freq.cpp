/**
 * Computes the most frequently occuring words in the input. Words in this
 * context are fixed-size chunks of characters.
 */
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "InputGen.h"

char normalize(char c) {
  if (c >= '0' && c <= '9') {
    // Map all int digits to 0
    return '0';
  }

  return c;
}

void normalize(char *c, int len) {
  for (int i = 0; i < len; i++) {
    c[i] = normalize(c[i]);
  }
}

using WordType = uint32_t;
constexpr int WORD_LEN = sizeof(WordType);

int main(int argc, char **argv) {

  std::unordered_map<WordType, int> wordCount;

  if (argc == 1) {
    std::string expr;
    while (std::getline(std::cin, expr)) {
      for (int i = 0; (i + WORD_LEN) < expr.size(); i += WORD_LEN) {
        char word[WORD_LEN];
        std::memcpy(word, expr.data() + i, WORD_LEN);
        normalize(word, WORD_LEN);
        WordType k;
        std::memcpy(&k, word, sizeof(k));
        wordCount[k]++;
      }
    }
  } else {
    // Generate some input ourselves
    std::cerr << "Generating input\n";
    InputGen gen;
    for (int nExpr = 0; nExpr < 100; nExpr++) {
      auto [expr, val] = gen.genExpr(50'000);
      for (int i = 0; i < expr.size(); i += WORD_LEN) {
        char word[WORD_LEN];
        std::memcpy(word, expr.data() + i, WORD_LEN);
        normalize(word, WORD_LEN);
        WordType k;
        std::memcpy(&k, word, sizeof(k));
        wordCount[k]++;
      }
    }
  }

  std::vector<std::pair<int, WordType>> sorted;
  for (auto [k, count] : wordCount) {
    sorted.emplace_back(count, k);
  }

  std::sort(sorted.begin(), sorted.end(),
            [](auto lhs, auto rhs) { return lhs.first > rhs.first; });

  // Report findings
  for (auto [count, k] : sorted) {
    std::string word(WORD_LEN, '0');
    std::memcpy(word.data(), &k, WORD_LEN);
    std::cerr << word << ": " << count << "\n";
  }
  return 0;
}
