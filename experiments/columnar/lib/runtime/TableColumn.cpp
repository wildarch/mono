#include <cassert>
#include <cstdlib>
#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/types.h>

#include "columnar/runtime/ByteArray.h"
#include "columnar/runtime/PipelineContext.h"
#include "columnar/runtime/TableColumn.h"

namespace columnar::runtime {

TableColumn::TableColumn(int idx) : _idx(idx) {}

void TableColumn::open(const std::string &path) {
  _reader = parquet::ParquetFileReader::OpenFile(path);
}

void TableColumn::close() { _reader->Close(); }

void TableColumn::read(int rowGroup, int skip, std::int64_t size,
                       std::int32_t *buffer) {
  // TODO: avoid such reads in IR generation.
  if (size == 0) {
    return;
  }

  auto groupReader = _reader->RowGroup(rowGroup);
  auto colReader = groupReader->Column(_idx);

  assert(colReader->type() == parquet::Type::INT32);
  auto *i32Reader = static_cast<parquet::Int32Reader *>(colReader.get());

  i32Reader->Skip(skip);

  std::int64_t valuesRead;
  i32Reader->ReadBatch(size, nullptr, nullptr, buffer, &valuesRead);
  assert(valuesRead == size);
}

void TableColumn::read(PipelineContext &ctx, int rowGroup, int skip,
                       std::int64_t size, ByteArray *buffer) {
  // TODO: avoid such reads in IR generation.
  if (size == 0) {
    return;
  }

  auto groupReader = _reader->RowGroup(rowGroup);
  auto colReader = groupReader->Column(_idx);

  assert(colReader->type() == parquet::Type::BYTE_ARRAY);
  auto *baReader = static_cast<parquet::ByteArrayReader *>(colReader.get());

  baReader->Skip(skip);

  // TODO: Avoid allocating a buffer here.
  llvm::SmallVector<parquet::ByteArray, 1024> tmp(size);

  std::int64_t valuesRead;
  baReader->ReadBatch(size, nullptr, nullptr, tmp.data(), &valuesRead);
  assert(valuesRead == size);

  // Convert to the format we use internally.
  for (auto [i, src] : llvm::enumerate(tmp)) {
    buffer[i] = ByteArray(src.ptr, src.len, ctx.allocator());
  }
}

} // namespace columnar::runtime
