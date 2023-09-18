#pragma once

#include <cstdint>

namespace execution {

class ColumnIdx {
private:
  uint16_t _val;

public:
  inline ColumnIdx(uint16_t val) : _val(val) {}

  inline uint16_t value() const { return _val; }
};

} // namespace execution