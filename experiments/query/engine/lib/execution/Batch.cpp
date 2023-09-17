#include "execution/Batch.h"

namespace execution {

Batch::Column::Column(PhysicalColumnType type, uint32_t rows)
    : _type(type), _data(rows * physicalColumnTypeSize(type)) {}

void Batch::Column::resize(uint32_t rows) {
  _data.resize(rows * physicalColumnTypeSize(_type));
}

Batch::Batch(std::span<const PhysicalColumnType> columnTypes, uint32_t rows)
    : _rows(rows) {
  for (const auto &type : columnTypes) {
    _columns.emplace_back(type, rows);
  }
}

void Batch::setRows(uint32_t rows) {
  _rows = rows;
  for (auto &c : _columns) {
    c.resize(rows);
  }
}

} // namespace execution