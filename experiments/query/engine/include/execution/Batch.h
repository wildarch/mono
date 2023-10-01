#pragma once

#include "execution/Common.h"
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace execution {

enum class PhysicalColumnType {
  INT32,
  INT64,
  DOUBLE,
  STRING,
};

template <PhysicalColumnType type> struct StoredType;

template <> struct StoredType<PhysicalColumnType::INT32> {
  using type = int32_t;
};

template <> struct StoredType<PhysicalColumnType::INT64> {
  using type = int64_t;
};

template <> struct StoredType<PhysicalColumnType::DOUBLE> {
  using type = double;
};

template <> struct StoredType<PhysicalColumnType::STRING> {
  using type = SmallString;
};

template <PhysicalColumnType t> using CType = StoredType<t>::type;

constexpr auto physicalColumnTypeSize(PhysicalColumnType t) {
  switch (t) {
  case PhysicalColumnType::INT32:
    return sizeof(int32_t);
  case PhysicalColumnType::INT64:
    return sizeof(int64_t);
  case PhysicalColumnType::DOUBLE:
    return sizeof(double);
  case PhysicalColumnType::STRING:
    return sizeof(SmallString);
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
CASE(INT64, int64_t)
CASE(DOUBLE, double)
CASE(STRING, SmallString)

#undef CASE

} // namespace execution