#include <cstdint>

#include "columnar/Runtime.h"

namespace {

using TableId = std::uint64_t;
using ColumnId = std::uint64_t;

class TableScanner;
class TableColumn;
class Printer;
class PrintChunk;

struct MemRef {
  void *alloc;
  void *align;
  std::size_t offset;
  std::size_t size;
  std::size_t stride;
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

TableScanner *col_table_scanner_open(TableId id) {
  llvm::errs() << "col_table_scanner_open\n";
  // TODO
  return nullptr;
}

struct ClaimedRange {
  std::size_t start;
  std::size_t size;
};

ClaimedRange col_table_scanner_claim_chunk(TableScanner *scanner) {
  llvm::errs() << "col_table_scanner_claim_chunk\n";
  return {0, 0};
}

TableColumn *col_table_column_open(TableId table, ColumnId column) {
  llvm::errs() << "col_table_column_open\n";
  return nullptr;
}

void col_table_column_read(TableColumn *column, std::size_t start,
                           std::size_t size,
                           // TODO: decode this memref
                           MEMREF_PARAM(col)) {
  MEMREF_VAR(col);
  llvm::errs() << "col_table_column_read\n";
  llvm::errs() << "allocated=" << col.alloc << "\n";
  llvm::errs() << "aligned=" << col.align << "\n";
  llvm::errs() << "offset=" << col.offset << "\n";
  llvm::errs() << "size=" << col.size << "\n";
  llvm::errs() << "stride=" << col.stride << "\n";
  // TODO
}

Printer *col_print_open() {
  llvm::errs() << "col_print_open\n";
  return nullptr;
}

void col_print_write(Printer *printer, PrintChunk *chunk) {
  llvm::errs() << "col_print_write\n";
  // TODO
}

PrintChunk *col_print_chunk_alloc(std::size_t size) {
  llvm::errs() << "col_print_chunk_alloc\n";
  return nullptr;
}

void col_print_chunk_append(PrintChunk *chunk, MEMREF_PARAM(col),
                            MEMREF_PARAM(sel)) {
  MEMREF_VAR(col);
  MEMREF_VAR(sel);
  llvm::errs() << "col_print_chunk_append\n";
  // TODO
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
  REGISTER(col_table_column_read);
  REGISTER(col_print_open);
  REGISTER(col_print_write);
  REGISTER(col_print_chunk_alloc);
  REGISTER(col_print_chunk_append);

#undef REGISTER

  return map;
}

} // namespace columnar