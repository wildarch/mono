#include "parse/Lexer.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

static dblang::LogicalResult readFileToString(const std::string &filename,
                                              std::string &out) {
  std::ifstream file{filename};
  if (!file.is_open()) {
    return dblang::reportError(dblang::Loc{filename}, "could not open file");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  if (file.bad()) {
    return dblang::reportError(dblang::Loc{filename}, "error reading file");
  }

  out = buffer.str();
  return dblang::LogicalResult::success();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <source file>\n";
    return 1;
  }

  std::string sourceFilename{argv[1]};
  // Read the source file
  std::string sourceContents;
  if (dblang::failed(readFileToString(sourceFilename, sourceContents))) {
    return 1;
  }

  // Source file -> tokens
  std::vector<dblang::Token> tokens;
  if (dblang::failed(dblang::lex(sourceFilename, sourceContents, tokens))) {
    return 1;
  }
}