/**
 * Enumerates all possible fixed-length sequences of characters that may appear
 * in the input. This is used to determine the cases needed in an interpreter
 * dispatch table.
 */
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "LegalWords.h"
#include "hash32.h"

static void countLegalWordsUpTo(std::size_t limit) {
  for (std::size_t i = 0; i < limit; i++) {
    LegalWords legalWords(i);
    auto nwords = legalWords.getAll().size();
    std::cout << "word len " << i << " number of legal words: " << nwords
              << "\n";
  }
}

static void printLegalWords(std::size_t wordLen) {
  LegalWords legalWords(wordLen);
  for (auto w : legalWords.getAll()) {
    std::cout << std::string_view(w.data(), w.size()) << "\n";
  }
}

// Check how many duplicates we get when hashing 4-byte words using a simple and
// fast hashing function.
static void hashLegalWords() {
  constexpr int SLOTS = 256;
  std::array<int, SLOTS> slots;
  std::fill(slots.begin(), slots.end(), 0);

  LegalWords legalWords(4);
  for (const auto &word : legalWords.getAll()) {
    uint32_t wInt;
    std::memcpy(&wInt, word.data(), 4);

    auto h = murmurhash32_mix32(wInt);
    auto mask = SLOTS - 1;
    slots[h & mask]++;
  }

  int dups = 0;
  for (int i = 0; i < SLOTS; i++) {
    if (slots[i] > 1) {
      dups++;
      std::cout << "slot " << i << ": " << slots[i] << "\n";
    }
  }

  std::cout << "duplicates: " << dups << "\n";
}

int main(int argc, char **argv) {
  // printLegalWords(4);
  hashLegalWords();
  return 0;
}
