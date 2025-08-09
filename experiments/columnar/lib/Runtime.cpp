#include <cstdint>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include "columnar/Runtime.h"
#include "columnar/runtime/ByteArray.h"
#include "columnar/runtime/Hash.h"
#include "columnar/runtime/PipelineContext.h"
#include "columnar/runtime/Print.h"
#include "columnar/runtime/Scatter.h"
#include "columnar/runtime/TableColumn.h"
#include "columnar/runtime/TableScanner.h"
#include "columnar/runtime/TupleBuffer.h"

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

  template <typename T> llvm::MutableArrayRef<T> asMutableArrayRef() const {
    assert(stride == 1);
    return {static_cast<T *>(align) + offset, size};
  }

  template <typename T> T *asPointerMut() {
    assert(stride == 1);
    return static_cast<T *>(align) + offset;
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

TableScanner *col_table_scanner_open(const char *path) {
  auto *scanner = new TableScanner();

  // TODO: handle exception
  std::string pathString(path);
  scanner->open(pathString);

  return scanner;
}

void col_table_scanner_claim_chunk(TableScanner *scanner,
                                   std::int32_t *rowGroup, std::int32_t *skip,
                                   std::size_t *size) {
  auto claim = scanner->claimChunk();
  *rowGroup = claim.rowGroup;
  *skip = claim.skip;
  *size = claim.size;
}

TableColumn *col_table_column_open(const char *path, std::int32_t idx) {
  auto *col = new TableColumn(idx);
  std::string pathString(path);
  col->open(pathString);

  return col;
}

void col_table_column_read_int32(TableColumn *column, std::int32_t rowGroup,
                                 std::int32_t skip, std::int64_t size,
                                 MEMREF_PARAM(dest)) {
  MEMREF_VAR(dest);
  column->read(rowGroup, skip, size, dest.asPointerMut<std::int32_t>());
}

void col_table_column_read_byte_array(TableColumn *column,
                                      std::int32_t rowGroup, std::int32_t skip,
                                      std::int64_t size, MEMREF_PARAM(dest),
                                      PipelineContext *ctx) {
  MEMREF_VAR(dest);
  column->read(*ctx, rowGroup, skip, size, dest.asPointerMut<ByteArray>());
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

void col_print_chunk_append_string(PrintChunk *chunk, MEMREF_PARAM(col),
                                   MEMREF_PARAM(sel)) {
  MEMREF_VAR(col);
  MEMREF_VAR(sel);
  chunk->append(col.asArrayRef<char *>(), sel.asArrayRef<std::size_t>());
}

void col_hash_int64(MEMREF_PARAM(value), MEMREF_PARAM(sel),
                    MEMREF_PARAM(result)) {
  MEMREF_VAR(value);
  MEMREF_VAR(sel);
  MEMREF_VAR(result);
  Hash::hash(value.asArrayRef<std::uint64_t>(), sel.asArrayRef<std::size_t>(),
             result.asMutableArrayRef<std::uint64_t>());
}

TupleBufferGlobal *col_tuple_buffer_init() { return new TupleBufferGlobal(); }
void col_tuple_buffer_destroy(TupleBufferGlobal *buffer) { delete buffer; }

TupleBufferLocal *col_tuple_buffer_local_alloc(std::uint64_t tupleSize,
                                               std::uint64_t tupleAlignment) {
  return new TupleBufferLocal(tupleSize, tupleAlignment);
}

Allocator *col_tuple_buffer_local_get_allocator(TupleBufferLocal *buffer) {
  return buffer->allocator();
}

void col_tuple_buffer_local_insert(TupleBufferLocal *buffer,
                                   MEMREF_PARAM(hashes), MEMREF_PARAM(result)) {
  MEMREF_VAR(hashes);
  MEMREF_VAR(result);
  buffer->insert(hashes.asArrayRef<hash64_t>(),
                 result.asMutableArrayRef<void *>());
}

void col_scatter_byte_array(MEMREF_PARAM(sel), MEMREF_PARAM(value),
                            MEMREF_PARAM(dest), Allocator *allocator) {
  MEMREF_VAR(sel);
  MEMREF_VAR(value);
  MEMREF_VAR(dest);
  scatterByteArray(sel.asArrayRef<std::size_t>(), value.asArrayRef<ByteArray>(),
                   value.asArrayRef<ByteArray *>(), *allocator);
}

void col_scatter_int32(MEMREF_PARAM(sel), MEMREF_PARAM(value),
                       MEMREF_PARAM(dest)) {
  MEMREF_VAR(sel);
  MEMREF_VAR(value);
  MEMREF_VAR(dest);
  scatterPrimitive(sel.asArrayRef<std::size_t>(),
                   value.asArrayRef<std::uint32_t>(),
                   dest.asArrayRef<std::uint32_t *>());
}

void col_tuple_buffer_merge(TupleBufferGlobal *global,
                            TupleBufferLocal *local) {
  global->merge(*local);
}

void col_debug_i32(std::int32_t v) { llvm::errs() << "DEBUG: " << v << "\n"; }
void col_debug_i64(std::int64_t v) { llvm::errs() << "DEBUG: " << v << "\n"; }
} // extern "C"

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
  REGISTER(col_table_column_read_byte_array);
  REGISTER(col_print_open);
  REGISTER(col_print_write);
  REGISTER(col_print_chunk_alloc);
  REGISTER(col_print_chunk_append_int32);
  REGISTER(col_print_chunk_append_string);
  REGISTER(col_hash_int64);
  REGISTER(col_tuple_buffer_init);
  REGISTER(col_tuple_buffer_destroy);
  REGISTER(col_tuple_buffer_local_alloc);
  REGISTER(col_tuple_buffer_local_get_allocator);
  REGISTER(col_tuple_buffer_local_insert);
  REGISTER(col_scatter_byte_array);
  REGISTER(col_scatter_int32);
  REGISTER(col_tuple_buffer_merge);
  REGISTER(col_debug_i32);
  REGISTER(col_debug_i64);

#undef REGISTER

  return map;
}

} // namespace columnar
