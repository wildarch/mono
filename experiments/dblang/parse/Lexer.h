#pragma once

#include "util/Result.h"
#include <cassert>
#include <string_view>
#include <vector>

namespace dblang {

#define GRAPHALG_ENUM_TOKEN_KIND(XX)                                           \
  XX(INVALID)     /* SPECIAL: For inputs that could not be tokenized. */       \
  XX(END_OF_FILE) /* SPECIAL: End of file marker. */                           \
  XX(IDENT)                                                                    \
  XX(INT)                                                                      \
  XX(FLOAT)                                                                    \
  /* Keywords */                                                               \
  /* Punctuation */                                                            \
  /* Operators */                                                              \
  /* Literals */                                                               \
  XX(TRUE)                                                                     \
  XX(FALSE)

struct Token {
  enum Kind {
#define GA_CASE(X) X,
    GRAPHALG_ENUM_TOKEN_KIND(GA_CASE)
#undef GA_CASE
  };

  static const char *kindName(Kind k);

  Kind type;
  // TODO: source location
  std::string_view body = "";
};

LogicalResult lex(std::string_view filename, std::string_view source,
                  std::vector<Token> &tokens);

} // namespace dblang