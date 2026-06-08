#include "parse/Lexer.h"
#include "parse/Parser.h"
#include "util/FileSystem.h"
#include "util/Result.h"
#include <iostream>
#include <vector>

using namespace dblang;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <source file>\n";
    return 1;
  }

  std::string sourceFilename{argv[1]};
  // Read the source file
  std::string sourceContents;
  if (failed(readFileToString(sourceFilename, sourceContents))) {
    return 1;
  }

  // Source file -> tokens
  std::vector<Token> tokens;
  if (failed(lex(sourceFilename, sourceContents, tokens))) {
    return 1;
  }

  for (const auto &token : tokens) {
    std::cout << token << "\n";
  }

  if (failed(parse(tokens))) {
    return 1;
  }
}