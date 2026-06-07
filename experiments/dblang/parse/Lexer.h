#pragma once

#include "parse/Location.h"
#include "util/Result.h"
#include <cassert>
#include <string_view>
#include <vector>

namespace dblang {

#define DBLANG_ENUM_TOKEN_KIND(XX)                                             \
  XX(INVALID)     /* SPECIAL: For inputs that could not be tokenized. */       \
  XX(END_OF_FILE) /* SPECIAL: End of file marker. */                           \
  XX(IDENT)                                                                    \
  XX(INT)                                                                      \
  XX(FLOAT)                                                                    \
  XX(STRING)                                                                   \
  XX(CHAR)                                                                     \
  /* Keywords */                                                               \
  XX(BREAK) /* break */                                                        \
  /* Punctuation */                                                            \
  XX(LPAREN)    /* ( */                                                        \
  XX(RPAREN)    /* ) */                                                        \
  XX(LBRACE)    /* { */                                                        \
  XX(RBRACE)    /* } */                                                        \
  XX(LSBRACKET) /* [ */                                                        \
  XX(RSBRACKET) /* ] */                                                        \
  XX(LANGLE)    /* < */                                                        \
  XX(RANGLE)    /* > */                                                        \
  XX(COLON)     /* : */                                                        \
  XX(COMMA)     /* , */                                                        \
  XX(DOT)       /* . */                                                        \
  XX(SEMI)      /* ; */                                                        \
  /* Operators */                                                              \
  XX(PLUS)        /* + */                                                      \
  XX(MINUS)       /* - */                                                      \
  XX(ASTERISK)    /* * */                                                      \
  XX(SLASH)       /* / */                                                      \
  XX(ASSIGN)      /* = */                                                      \
  XX(EXCLAMATION) /* ! */                                                      \
  XX(TILDE)       /* ~ */                                                      \
  XX(AMPERSAND)   /* & */                                                      \
  XX(PIPE)        /* | */                                                      \
  XX(QMARK)       /* ? */                                                      \
  XX(PERCENT)     /* % */                                                      \
  XX(CARET)       /* ^ */                                                      \
  XX(ARROW)       /* -> */                                                     \
  XX(DAMPERSAND)  /* && */                                                     \
  XX(DPIPE)       /* || */                                                     \
  XX(EQUAL)       /* == */                                                     \
  XX(NOT_EQUAL)   /* != */                                                     \
  XX(LEQ)         /* <= */                                                     \
  XX(GEQ)         /* >= */                                                     \
  XX(INC)         /* ++ */                                                     \
  XX(DEC)         /* -- */                                                     \
  XX(LSHIFT)      /* << */                                                     \
  XX(RSHIFT)      /* >> */

struct Token {
  enum Kind {
#define CASE(X) X,
    DBLANG_ENUM_TOKEN_KIND(CASE)
#undef CASE
  };

  static const char *kindName(Kind k);

  Loc loc;
  Kind kind;
  std::string_view body = "";
};

std::ostream &operator<<(std::ostream &os, const Token &token);

LogicalResult lex(std::string_view filename, std::string_view source,
                  std::vector<Token> &tokens);

} // namespace dblang