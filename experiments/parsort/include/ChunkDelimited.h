#pragma once

#include <string_view>
#include <vector>

/**
 * Splits a string containing multiple lines into chunks that can be read in
 * parallel.
 */
class ChunkDelimited {
public:
  ChunkDelimited(std::string_view input, std::size_t chunks) {
    auto chunkSize = input.size() / chunks;
    for (int i = 0; i < chunks; i++) {
      _chunks.push_back(input.begin() + i * chunkSize);
    }

    _chunks.push_back(input.end());
  }

  template <typename F> void visitLinesInChunk(std::size_t chunk, F &&f) {
    bool isFirst = chunk == 0;
    bool isLast = chunk == (_chunks.size() - 1);

    const char *start = _chunks[chunk];
    const char *end = _chunks[chunk + 1];
    std::string_view line(start, std::distance(start, end));

    if (!isFirst) {
      // Find the first newline
      while (start < end && *start != '\n') {
        start++;
      }

      if (start == end || *start != '\n') {
        // No newline within the buffer, skip.
        return;
      }

      // Skip over the start of the first line.
      start++;
    }

    // NOTE: start now points to the start of the next line.
    do {
      // Find next newline
      const char *lineEnd = start;
      while (lineEnd < end && *lineEnd != '\n') {
        lineEnd++;
      }

      if (lineEnd == end && !isLast) {
        // Reached the end of our part without finding a newline.
        // Keep scanning in the following parts until we find a newline.
        while (lineEnd < _chunks.back() && *lineEnd != '\n') {
          lineEnd++;
        }
      }

      std::string_view line(start, std::distance(start, lineEnd));
      f(line);
      start = lineEnd + 1;
    } while (start < end);
  }

private:
  std::vector<const char *> _chunks;
};
