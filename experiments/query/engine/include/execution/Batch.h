#pragma once

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace execution {

enum class PhysicalColumnType {
  INT32,
  DOUBLE,
  STRING_PTR,
};

class StringPtr {
  const char *_ptr;
};

template <PhysicalColumnType type> struct StoredType;

template <> struct StoredType<PhysicalColumnType::INT32> {
  using type = int32_t;
};

template <> struct StoredType<PhysicalColumnType::DOUBLE> {
  using type = double;
};

template <> struct StoredType<PhysicalColumnType::STRING_PTR> {
  using type = StringPtr;
};

template <PhysicalColumnType t> using CType = StoredType<t>::type;

constexpr auto physicalColumnTypeSize(PhysicalColumnType t) {
  switch (t) {
  case PhysicalColumnType::INT32:
    return sizeof(int32_t);
  case PhysicalColumnType::DOUBLE:
    return sizeof(double);
  case PhysicalColumnType::STRING_PTR:
    return sizeof(StringPtr);
  default:
    __builtin_unreachable();
  }
}

class Batch {
public:
  class Column {
    friend class Batch;

  private:
    PhysicalColumnType _type;
    // TODO: use __m256i
    std::vector<std::byte> _data;

  public:
    Column(PhysicalColumnType type, uint32_t rows);

    inline auto type() const { return _type; }

    template <PhysicalColumnType type>
    std::span<const typename StoredType<type>::type> get() const {
      assert(type == _type);
      using T = typename StoredType<type>::type;
      auto *ptr = reinterpret_cast<const T *>(_data.data());
      size_t numElements = _data.size() / sizeof(T);
      return std::span(ptr, ptr + numElements);
    }

    template <PhysicalColumnType type> auto *getForWrite();

  private:
    void resize(uint32_t rows);
  };

private:
  uint32_t _rows;
  std::vector<Column> _columns;

public:
  Batch(std::span<const PhysicalColumnType> columnTypes, uint32_t rows);

  void setRows(uint32_t rows);

  inline auto rows() const { return _rows; }
  inline const auto &columns() const { return _columns; }
  inline auto &columnsForWrite() { return _columns; }
};

#define CASE(ptype, ctype)                                                     \
  template <>                                                                  \
  inline auto *Batch::Column::getForWrite<PhysicalColumnType::ptype>() {       \
    assert(_type == PhysicalColumnType::ptype);                                \
    return reinterpret_cast<ctype *>(_data.data());                            \
  }

CASE(INT32, int32_t)
CASE(DOUBLE, double)
CASE(STRING_PTR, StringPtr)

#undef CASE

} // namespace execution