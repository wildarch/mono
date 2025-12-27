#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "hash32.h"

struct State {
  std::uint16_t intValue = 0;
  bool negInt = false;

  std::vector<std::int64_t> output;
  std::vector<char> ops;
};

char CASES_ATOM_LHS[] = {"-000"
                         "0000"
                         "(-00"
                         "(000"};

template <typename F> void enumerate(char *input, int idx, F &&f) {
  if (idx == 4) {
    f(input);
    return;
  }

  if (input[idx] != '0') {
    // Nothing to enumerate here.
    enumerate(input, idx + 1, f);
    return;
  }

  for (char c = '0'; c <= '9'; c++) {
    input[idx] = c;
    enumerate(input, idx + 1, f);
  }

  input[idx] = '0';
}

int main(int argc, char **argv) {
  int table[128];
  std::memset(table, 0, sizeof(table));

  {
    // 0000
    char buf[] = "0000";
    enumerate(buf, 0, [&](char *word) {
      std::uint32_t w;
      std::memcpy(&w, word, sizeof(w));
      auto h = murmurhash32_mix32(w) % 128;
      auto &bucket = table[h];
      if (bucket == 0) {
        // Newly used.
        bucket = 1;
      } else if (bucket == 1) {
        // Already set.
      } else if (bucket == 1000) {
        // Already a collision.
      } else {
        // Collision
        bucket = 1000;
      }
    });
  }

  {
    // -000
    char buf[] = "-000";
    enumerate(buf, 0, [&](char *word) {
      std::uint32_t w;
      std::memcpy(&w, word, sizeof(w));
      auto h = murmurhash32_mix32(w) % 128;
      auto &bucket = table[h];
      if (bucket == 0) {
        // Newly used.
        bucket = 2;
      } else if (bucket == 2) {
        // Already set.
      } else if (bucket == 1000) {
        // Already a collision.
      } else {
        // Collision
        bucket = 1000;
      }
    });
  }

  for (int i = 0; i < 128; i++) {
    if (table[i] == 1000) {
      std::cout << "collision at " << i << "\n";
    }
  }
}
