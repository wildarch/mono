#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace execution {

class ColumnIdx {
private:
  uint16_t _val;

public:
  inline ColumnIdx(uint16_t val) : _val(val) {}

  inline uint16_t value() const { return _val; }
};

#pragma pack(push, 1)
class SmallString {
private:
  static constexpr uint32_t MAX_INLINE_LEN = 12;
  static constexpr size_t PREFIX_LEN = 4;
  // Based on https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf
  uint32_t _len;
  union {
    // The full string, if it fits in 12 bytes.
    char _inline[MAX_INLINE_LEN];
    // Prefix and pointer to the full string, if longer than 12 bytes
    struct {
      char _prefix[PREFIX_LEN];
      uint8_t *_ptr;
    };
  };

  void cloneFrom(const SmallString &other) {
    // deallocate old string if needed
    if (_len > MAX_INLINE_LEN) {
      std::free(_ptr);
    }

    _len = other._len;
    if (_len <= MAX_INLINE_LEN) {
      std::memcpy(_inline, other._inline, other._len);
    } else {
      _ptr = static_cast<uint8_t *>(std::malloc(other._len));
      assert(_ptr != nullptr);

      assert(other._len >= PREFIX_LEN);
      std::memcpy(_prefix, other._ptr, PREFIX_LEN);
      std::memcpy(_ptr, other._ptr, other._len);
    }
  }

public:
  inline SmallString(uint32_t len, const uint8_t *ptr) : _len(len) {
    if (len <= MAX_INLINE_LEN) {
      std::memcpy(_inline, ptr, len);
    } else {

      _ptr = static_cast<uint8_t *>(std::malloc(len));
      assert(_ptr != nullptr);

      assert(len >= PREFIX_LEN);
      std::memcpy(_prefix, ptr, PREFIX_LEN);
      std::memcpy(_ptr, ptr, len);
    }
  }
  inline SmallString(const SmallString &other) { cloneFrom(other); }
  inline SmallString &operator=(const SmallString &other) {
    cloneFrom(other);
    return *this;
  }
  SmallString(SmallString &&) = delete;

  inline ~SmallString() {
    if (_len > MAX_INLINE_LEN) {
      std::free(_ptr);
    }
  }

  constexpr operator std::string_view() const {
    if (_len <= MAX_INLINE_LEN) {
      return std::string_view(_inline, _len);
    } else {
      return std::string_view(_prefix, _len);
    }
  }
};
#pragma pack(pop)

std::ostream &operator<<(std::ostream &os, const SmallString &s);

static_assert(sizeof(SmallString) == 16);

} // namespace execution