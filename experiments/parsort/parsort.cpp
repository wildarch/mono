#include <charconv>
#include <cstdint>
#include <iostream>
#include <string_view>

// for mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct Parser {
  static constexpr std::size_t NUM_PARTS = 16;
  std::vector<const char *> parts;

  // TODO: parse method.
};

int main(int argc, char **argv) {
  // Check we have exactly one argument
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <edges file>\n";
    return 1;
  }

  // Open the file for reading
  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    perror("Error: Cannot open file");
    return 1;
  }

  // Get the file size
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("Error: Cannot get file size");
    close(fd);
    return 1;
  }
  size_t file_size = sb.st_size;

  if (file_size == 0) {
    std::cerr << "Error: Empty input file\n";
    close(fd);
    return 0;
  }

  // Mmap the file
  char *file_contents =
      static_cast<char *>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (file_contents == MAP_FAILED) {
    perror("Error: Cannot mmap file");
    close(fd);
    return 1;
  }
  close(fd);

  Parser parser;
  auto part_size = file_size / Parser::NUM_PARTS;
  for (int i = 0; i < Parser::NUM_PARTS; i++) {
    parser.parts.push_back(file_contents + i * part_size);
  }

  // TODO: parser.parse()

  // Process the mmapped data
  const char *end_ptr = file_contents + file_size;
  const char *current_ptr = file_contents;
  while (current_ptr < end_ptr) {
    const char *line_start_ptr =
        current_ptr; // Keep track of line start for error message
    const char *line_end_ptr = current_ptr;
    while (line_end_ptr < end_ptr && *line_end_ptr != '\n') {
      line_end_ptr++;
    }

    std::string_view line_view(line_start_ptr, line_end_ptr - line_start_ptr);

    if (!line_view.empty()) {
      uint64_t u, v;
      double w;

      const char *parse_begin = line_view.data();
      const char *parse_end = line_view.data() + line_view.length();

      // Parse u
      auto [ptr_u, ec_u] = std::from_chars(parse_begin, parse_end, u);
      if (ec_u != std::errc() || ptr_u == parse_begin) {
        std::cerr << "Warning: Could not parse u in line: " << line_view
                  << "\n";
        current_ptr = line_end_ptr + 1;
        continue;
      }

      // Skip space after u
      if (ptr_u < parse_end && *ptr_u == ' ') {
        ptr_u++;
      } else {
        std::cerr << "Warning: Expected space after u in line: " << line_view
                  << "\n";
        current_ptr = line_end_ptr + 1;
        continue;
      }

      // Parse v
      auto [ptr_v, ec_v] = std::from_chars(ptr_u, parse_end, v);
      if (ec_v != std::errc() || ptr_v == ptr_u) {
        std::cerr << "Warning: Could not parse v in line: " << line_view
                  << "\n";
        current_ptr = line_end_ptr + 1;
        continue;
      }

      // Skip space after v
      if (ptr_v < parse_end && *ptr_v == ' ') {
        ptr_v++;
      } else {
        std::cerr << "Warning: Expected space after v in line: " << line_view
                  << "\n";
        current_ptr = line_end_ptr + 1;
        continue;
      }

      // Parse w (double)
      auto [ptr_w, ec_w] = std::from_chars(ptr_v, parse_end, w);
      if (ec_w != std::errc() || ptr_w == ptr_v) {
        std::cerr << "Warning: Could not parse w (double) in line: "
                  << line_view << "\n";
        current_ptr = line_end_ptr + 1;
        continue;
      }

      // Ensure no extra characters after parsing w, except whitespace
      // Note: from_chars for floating point does not handle trailing whitespace
      // automatically
      while (ptr_w < parse_end && (*ptr_w == ' ' || *ptr_w == '\t')) {
        ptr_w++;
      }
      if (ptr_w < parse_end) {
        std::cerr << "Warning: Extra characters after w in line: " << line_view
                  << "\n";
        current_ptr = line_end_ptr + 1;
        continue;
      }

      // Print the parsed lines to the console
      std::cout << u << " " << v << " " << w << "\n";
    }
    current_ptr = line_end_ptr + 1;
  }

  // Unmap the file
  if (munmap(file_contents, file_size) == -1) { // Added error check for munmap
    perror("Error: Cannot munmap file");
  }

  return 0;
}
