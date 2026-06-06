#include "parse/Lexer.h"
#include "util/Result.h"
#include <vector>

int main(int argc, char **argv) {
  std::vector<dblang::Token> tokens;
  if (dblang::failed(dblang::lex("bla.c", "blabla", tokens))) {
    return 1;
  }
}