#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "columnar/runtime/Allocator.h"

namespace columnar::runtime {

class ByteArray {
private:
  std::byte *_data;

  ByteArray(const void *ptr, std::size_t len, Allocator &alloc)
      : _data(static_cast<std::byte *>(alloc.allocate(len + 1, 1))) {
    std::memcpy(_data, ptr, len);
    _data[len] = std::byte('\0');
  }

public:
  // Create from raw pointer.
  ByteArray(const std::byte *ptr, std::size_t len, Allocator &alloc)
      : ByteArray(static_cast<const void *>(ptr), len, alloc) {}
  ByteArray(const std::uint8_t *ptr, std::size_t len, Allocator &alloc)
      : ByteArray(static_cast<const void *>(ptr), len, alloc) {}
  // Copy into different allocator
  ByteArray(const ByteArray &arr, Allocator &alloc)
      : ByteArray(arr._data, arr.len(), alloc) {}

  std::size_t len() const {
    return std::strlen(reinterpret_cast<char *>(_data));
  }
  const std::byte *data() const { return _data; }
  std::string_view str() { return reinterpret_cast<char *>(_data); }
};

} // namespace columnar::runtime
