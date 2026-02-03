#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

// for mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ChunkDelimited.h"

static void handleLine(std::string_view line) {
  if (line.empty()) {
    return;
  }

  uint64_t u, v;
  double w;

  const char *parseBegin = line.data();
  const char *parseEnd = line.data() + line.length();

  // Parse u
  auto [uPtr, uEC] = std::from_chars(parseBegin, parseEnd, u);
  if (uEC != std::errc() || uPtr == parseBegin) {
    std::cerr << "warning: could not parse u in line: " << line << "\n";
    return;
  }

  // Skip space after u
  if (uPtr < parseEnd && *uPtr == ' ') {
    uPtr++;
  } else {
    std::cerr << "warning: expected space after u in line: " << line << "\n";
    return;
  }

  // Parse v
  auto [vPtr, vEC] = std::from_chars(uPtr, parseEnd, v);
  if (vEC != std::errc() || vPtr == uPtr) {
    std::cerr << "warning: could not parse v in line: " << line << "\n";
    return;
  }

  // Skip space after v
  if (vPtr < parseEnd && *vPtr == ' ') {
    vPtr++;
  } else {
    std::cerr << "warning: expected space after v in line: " << line << "\n";
    return;
  }

  // Parse w (double)
  auto [wPtr, wEC] = std::from_chars(vPtr, parseEnd, w);
  if (wEC != std::errc() || wPtr == vPtr) {
    std::cerr << "warning: could not parse w (double) in line: " << line
              << "\n";
    return;
  }

  // Ensure no extra characters after parsing w, except whitespace
  while (wPtr < parseEnd && (*wPtr == ' ' || *wPtr == '\t')) {
    wPtr++;
  }

  if (wPtr < parseEnd) {
    std::cerr << "warning: Extra characters after w in line: " << line << "\n";
    return;
  }

  // Print the parsed lines to the console
  std::cout << u << " " << v << " " << w << "\n";
}

int main(int argc, char **argv) {
  // Check we have exactly one argument
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <edges file>\n";
    return 1;
  }

  // Open the file for reading
  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    perror("error: Cannot open file");
    return 1;
  }

  // Get the file size
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("error: Cannot get file size");
    close(fd);
    return 1;
  }
  size_t fileSize = sb.st_size;

  if (fileSize == 0) {
    std::cerr << "error: Empty input file\n";
    close(fd);
    return 0;
  }

  // Mmap the file
  char *filePtr =
      static_cast<char *>(mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0));
  if (filePtr == MAP_FAILED) {
    perror("error: Cannot mmap file");
    close(fd);
    return 1;
  }
  close(fd);

  std::string_view fileContents(filePtr, fileSize);
  constexpr std::size_t NUM_PARTS = 16;
  ChunkDelimited chunkFile(fileContents, NUM_PARTS);

  for (int i = 0; i < NUM_PARTS; i++) {
    chunkFile.visitLinesInChunk(i, handleLine);
  }

  // Unmap the file
  if (munmap(filePtr, fileSize) == -1) {
    perror("error: Cannot munmap file");
  }

  return 0;
}
