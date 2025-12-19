/**
 * Enumerates all possible fixed-length sequences of characters that may appear
 * in the input. This is used to determine the cases needed in an interpreter
 * dispatch table.
 */
#include <iostream>

#include "LegalWords.h"

int main(int argc, char **argv) {
  /*
    for (std::size_t i = 0; i < 64; i++) {
      LegalWords legalWords(i);
      auto nwords = legalWords.getAll().size();
      std::cout << "word len " << i << " number of legal words: " << nwords
                << "\n";
    }
  */
  LegalWords legalWords(4);
  for (auto w : legalWords.getAll()) {
    std::cout << std::string_view(w.data(), w.size()) << "\n";
  }

  return 0;
}
