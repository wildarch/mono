#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

// for mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ChunkDelimited.h"

static void parseLine(std::ofstream &os, std::string_view line) {
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

  constexpr auto tupleSize = sizeof(u) + sizeof(v) + sizeof(w);
  char serialized[tupleSize];
  std::memcpy(serialized, &u, sizeof(u));
  std::memcpy(&serialized[sizeof(u)], &v, sizeof(v));
  std::memcpy(&serialized[sizeof(u) + sizeof(v)], &w, sizeof(w));
  os.write(serialized, tupleSize);
  if (os.bad()) {
    std::cerr << "error: failed to write tuple to buffer\n";
    return;
  }
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

  // Step 1: Split the input file into chunks.
  constexpr std::size_t NUM_CHUNKS = 16;
  ChunkDelimited chunkFile(fileContents, NUM_CHUNKS);

  // Step 2: Parse to binary format.
  auto tempDir = std::filesystem::temp_directory_path() / "parsort";
  std::error_code createError;
  std::filesystem::create_directories(tempDir, createError);
  if (createError) {
    std::cerr << "error: cannot create temp directory: "
              << createError.message() << "\n";
    return 1;
  }

  std::vector<std::filesystem::path> chunkPaths;
  for (int i = 0; i < NUM_CHUNKS; i++) {
    std::ostringstream filename;
    filename << "chunk" << std::setfill('0') << std::setw(4) << i;
    auto chunkFilePath = tempDir / filename.str();
    chunkPaths.push_back(chunkFilePath);
  }

  bool parseThreadsOk[NUM_CHUNKS];
  std::vector<std::thread> parseThreads;
  for (int i = 0; i < NUM_CHUNKS; i++) {
    parseThreadsOk[i] = false;
    parseThreads.emplace_back(
        [](int i, const ChunkDelimited &chunkFile,
           std::filesystem::path outputPath, bool &parseOk) {
          std::ofstream os(outputPath, std::ios_base::trunc);
          if (os.fail()) {
            std::cerr << "error opening file " << outputPath << '\n';
            return;
          }

          chunkFile.visitLinesInChunk(
              i, [&](std::string_view line) { parseLine(os, line); });
          parseOk = true;
        },
        i, std::ref(chunkFile), chunkPaths[i], std::ref(parseThreadsOk[i]));
  }

  bool parseOk = true;
  for (int i = 0; i < NUM_CHUNKS; i++) {
    parseThreads[i].join();
    parseOk &= parseThreadsOk[i];
  }

  std::cout << "parsed\n";

  // Unmap the file
  if (munmap(filePtr, fileSize) == -1) {
    perror("error: Cannot munmap file");
  }

  return parseOk ? 0 : 1;
}
