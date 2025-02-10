#include <cstdint>

#include "columnar/Runtime.h"

namespace {

using TableId = std::uint64_t;
using ColumnId = std::uint64_t;

class TableScanner;
class TableColumn;
class Printer;
class PrintChunk;

extern "C" {

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
                           void *ref0, void *ref1, std::size_t ref2,
                           std::size_t ref3, std::size_t ref4) {
  llvm::errs() << "col_table_column_read\n";
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

void col_print_chunk_append(PrintChunk *chunk,
                            // TODO: decode this memref
                            void *, void *, std::size_t, std::size_t,
                            std::size_t,
                            // TODO: decode this memref
                            void *, void *, std::size_t, std::size_t,
                            std::size_t) {
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