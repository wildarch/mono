#include "parse/Chunker.h"
#include "parse/Lexer.h"
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

  std::vector<Chunk> chunks;
  chunk(sourceFilename, sourceContents, chunks);

  std::vector<Token> tokens;
  for (const auto &chunk : chunks) {
    if (failed(lex(sourceContents, chunk.loc, tokens))) {
      return 1;
    }
  }

  // Source file -> tokens

  for (const auto &token : tokens) {
    std::cout << token << "\n";
  }

  /*
  if (failed(parse(tokens))) {
    return 1;
  }
  */
}