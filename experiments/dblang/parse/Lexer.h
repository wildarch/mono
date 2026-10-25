#pragma once

#include "parse/Location.h"
#include "util/Result.h"
#include <cassert>
#include <string_view>
#include <vector>

namespace dblang {

#define DBLANG_KEYWORDS(XX)                                                    \
  /* token kind, "keyword" */                                                  \
  XX(ALIAS, "alias")                                                           \
  XX(ALIGNOF, "alignof")                                                       \
  XX(BOOL, "bool")                                                             \
  XX(BREAK, "break")                                                           \
  XX(CASE, "case")                                                             \
  XX(CAST, "cast")                                                             \
  XX(CHAR_KW, "char")                                                          \
  XX(CONST, "const")                                                           \
  XX(CONTINUE, "continue")                                                     \
  XX(DEF, "def")                                                               \
  XX(DEFAULT, "default")                                                       \
  XX(DO, "do")                                                                 \
  XX(ELSE, "else")                                                             \
  XX(ENUM, "enum")                                                             \
  XX(FALSE, "false")                                                           \
  XX(FOR, "for")                                                               \
  XX(FUNC, "func")                                                             \
  XX(GLOBAL, "global")                                                         \
  XX(I8, "i8")                                                                 \
  XX(I16, "i16")                                                               \
  XX(I32, "i32")                                                               \
  XX(I64, "i64")                                                               \
  XX(I128, "i128")                                                             \
  XX(IF, "if")                                                                 \
  XX(IMPORT, "import")                                                         \
  XX(ISIZE, "isize")                                                           \
  XX(LET, "let")                                                               \
  XX(RETURN, "return")                                                         \
  XX(SIZEOF, "sizeof")                                                         \
  XX(STRUCT, "struct")                                                         \
  XX(SWITCH, "switch")                                                         \
  XX(TRUE, "true")                                                             \
  XX(U8, "u8")                                                                 \
  XX(U16, "u16")                                                               \
  XX(U32, "u32")                                                               \
  XX(U64, "u64")                                                               \
  XX(U128, "u128")                                                             \
  XX(USIZE, "usize")                                                           \
  XX(VOID, "void")                                                             \
  XX(WHILE, "while")

// All kinds that are not keywords
#define DBLANG_ENUM_TOKEN_KIND(XX)                                             \
  XX(INVALID)     /* SPECIAL: For inputs that could not be tokenized. */       \
  XX(END_OF_FILE) /* SPECIAL: End of file marker. */                           \
  XX(IDENT)                                                                    \
  XX(INT)                                                                      \
  XX(FLOAT)                                                                    \
  XX(STRING)                                                                   \
  XX(CHAR)                                                                     \
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
  XX(DCOLON)    /* :: */                                                       \
  XX(COMMA)     /* , */                                                        \
  XX(DOT)       /* . */                                                        \
  XX(SEMI)      /* ; */                                                        \
  /* Operators */                                                              \
  XX(PLUS)        /* + */                                                      \
  XX(MINUS)       /* - */                                                      \
  XX(ASTERISK)    /* * */                                                      \
  XX(SLASH)       /* / */                                                      \
  XX(PERCENT)     /* % */                                                      \
  XX(EXCLAMATION) /* ! */                                                      \
  XX(TILDE)       /* ~ */                                                      \
  XX(AMPERSAND)   /* & */                                                      \
  XX(PIPE)        /* | */                                                      \
  XX(CARET)       /* ^ */                                                      \
  XX(QMARK)       /* ? */                                                      \
  XX(ARROW)       /* -> */                                                     \
  XX(DPLUS)       /* ++ */                                                     \
  XX(DMINUS)      /* -- */                                                     \
  XX(DAMPERSAND)  /* && */                                                     \
  XX(DPIPE)       /* || */                                                     \
  XX(EQUAL)       /* == */                                                     \
  XX(NOT_EQUAL)   /* != */                                                     \
  XX(LEQ)         /* <= */                                                     \
  XX(GEQ)         /* >= */                                                     \
  XX(LSHIFT)      /* << */                                                     \
  XX(RSHIFT)      /* >> */                                                     \
  /* Assignment */                                                             \
  XX(ASSIGN)           /* = */                                                 \
  XX(ASTERISK_ASSIGN)  /* *= */                                                \
  XX(SLASH_ASSIGN)     /* /= */                                                \
  XX(PERCENT_ASSIGN)   /* %= */                                                \
  XX(PLUS_ASSIGN)      /* += */                                                \
  XX(MINUS_ASSIGN)     /* -= */                                                \
  XX(LSHIFT_ASSIGN)    /* <<= */                                               \
  XX(RSHIFT_ASSIGN)    /* >>= */                                               \
  XX(AMPERSAND_ASSIGN) /* &= */                                                \
  XX(CARET_ASSIGN)     /* ^= */                                                \
  XX(PIPE_ASSIGN)      /* |= */

struct Token {
  enum Kind {
#define CASE(X) X,
    DBLANG_ENUM_TOKEN_KIND(CASE)
#undef CASE
#define CASE(X, _Y) X,
        DBLANG_KEYWORDS(CASE)
#undef CASE
  };

  static const char *kindName(Kind k);

  Loc loc;
  Kind kind;
  std::string_view body = "";
};

std::ostream &operator<<(std::ostream &os, const Token &token);

LogicalResult lex(std::string_view source, Loc sourceLoc,
                  std::vector<Token> &tokens);

} // namespace dblang