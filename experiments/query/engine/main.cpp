#include <cstring>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "expected path to csv file\n";
    return 1;
  }

  std::ifstream infile(argv[1]);
  if (infile.fail()) {
    std::cerr << "cannot open csv file: " << strerror(errno) << "\n";
    return 1;
  }

  std::string line;
  while (!infile.eof()) {
    std::getline(infile, line);
    if (infile.fail()) {
      std::cerr << "error reading file: " << strerror(errno) << "\n";
    }

    std::cout << "line: " << line << "\n";
  }

  return 0;
}