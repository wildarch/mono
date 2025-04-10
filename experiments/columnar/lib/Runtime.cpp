#include <cstdint>
#include <iostream>

#include "columnar/Runtime.h"
#include "columnar/runtime/Print.h"
#include "columnar/runtime/TableColumn.h"
#include "columnar/runtime/TableScanner.h"

namespace {

using namespace columnar::runtime;

struct MemRef {
  void *alloc;
  void *align;
  std::size_t offset;
  std::size_t size;
  std::size_t stride;

  template <typename T> llvm::ArrayRef<T> asArrayRef() const {
    assert(stride == 1);
    return {static_cast<const T *>(align) + offset, size};
  }

  template <typename T> llvm::MutableArrayRef<T> asArrayRefMut() {
    assert(stride == 1);
    return {static_cast<T *>(align) + offset, size};
  }
};

#define MEMREF_PARAM(name)                                                     \
  void *name##_alloc, void *name##_align, std::size_t name##_offset,           \
      std::size_t name##_size, std::size_t name##_stride

#define MEMREF_VAR(name)                                                       \
  MemRef name {                                                                \
    name##_alloc, name##_align, name##_offset, name##_size, name##_stride,     \
  }

extern "C" {
/**
 * Note: I suspect the format for memref is:
 * - void* allocated pointer
 * - void* aligned pointer
 * - std::size_t offset
 * - std::size_t size
 * - std::size_t stride
 */

TableScanner *col_table_scanner_open(const char *path) {
  auto *col = new TableScanner();
  if (auto err = col->open(path)) {
    llvm::errs() << "Invalid table path '" << path << "': " << err << "\n";
    std::abort();
  }

  return col;
}

// NOTE: C repr instead of C++ (for interop).
struct ClaimedRange {
  std::size_t start;
  std::size_t size;
};

ClaimedRange col_table_scanner_claim_chunk(TableScanner *scanner) {
  auto claim = scanner->claimChunk();
  return {claim.start, claim.size};
}

TableColumn *col_table_column_open(const char *path) {
  auto *col = new TableColumn();
  if (auto err = col->open(path)) {
    llvm::errs() << "Invalid columnar path '" << path << "': " << err << "\n";
    std::abort();
  }

  return col;
}

void col_table_column_read_int32(TableColumn *column, std::size_t start,
                                 std::size_t size, MEMREF_PARAM(dest)) {
  MEMREF_VAR(dest);
  column->read(start, size, dest.asArrayRefMut<std::int32_t>());
}

Printer *col_print_open() { return new Printer(); }

void col_print_write(Printer *printer, PrintChunk *chunk) {
  printer->write(*chunk);
  delete chunk;
}

PrintChunk *col_print_chunk_alloc(std::size_t size) {
  return new PrintChunk(size);
}

void col_print_chunk_append_int32(PrintChunk *chunk, MEMREF_PARAM(col),
                                  MEMREF_PARAM(sel)) {
  MEMREF_VAR(col);
  MEMREF_VAR(sel);
  chunk->append(col.asArrayRef<std::int32_t>(), sel.asArrayRef<std::size_t>());
}
}

template <typename T>
static void registerSymbol(llvm::orc::SymbolMap &map,
                           llvm::orc::MangleAndInterner interner,
                           llvm::StringRef name, T func) {
  map[interner(name)] = {
      llvm::orc::ExecutorAddr::fromPtr(func),
      llvm::JITSymbolFlags::None,
  };
}

} // namespace

namespace columnar {

llvm::orc::SymbolMap registerRuntimeSymbols(llvm::orc::MangleAndInterner mai) {
  llvm::orc::SymbolMap map;

#define REGISTER(sym) registerSymbol(map, mai, #sym, &sym)

  REGISTER(col_table_scanner_open);
  REGISTER(col_table_scanner_claim_chunk);
  REGISTER(col_table_column_open);
  REGISTER(col_table_column_read_int32);
  REGISTER(col_print_open);
  REGISTER(col_print_write);
  REGISTER(col_print_chunk_alloc);
  REGISTER(col_print_chunk_append_int32);

#undef REGISTER

  return map;
}

} // namespace columnar