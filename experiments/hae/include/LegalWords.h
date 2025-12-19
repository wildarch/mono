#pragma once

#include <set>
#include <string_view>
#include <vector>
class LegalWords {
private:
  using Word = std::vector<char>;
  std::set<std::string_view> illegalWords;
  std::set<Word> legalWords;

  void buildIllegalWords();
  void enumerate(Word &word, std::size_t i);

public:
  LegalWords(std::size_t n);

  bool isLegal(std::string_view word);
  const auto &getAll() { return legalWords; }
};
