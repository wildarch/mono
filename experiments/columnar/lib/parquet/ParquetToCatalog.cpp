#include <filesystem>

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

#include <parquet/metadata.h>
#include <parquet/schema.h>

#include "columnar/Catalog.h"
#include "columnar/Columnar.h"

namespace columnar::parquet {

static mlir::Type convertPhysicalType(mlir::MLIRContext *ctx,
                                      ::parquet::Type::type type) {
  switch (type) {
  case ::parquet::Type::BOOLEAN:
    return mlir::IntegerType::get(ctx, 1);
  case ::parquet::Type::INT32:
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
  case ::parquet::Type::INT64:
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
  case ::parquet::Type::FLOAT:
    return mlir::Float32Type::get(ctx);
  case ::parquet::Type::DOUBLE:
    return mlir::Float64Type::get(ctx);
  case ::parquet::Type::INT96:
    llvm_unreachable("INT96 not supported");
  case ::parquet::Type::BYTE_ARRAY:
    return columnar::StringType::get(ctx);
  case ::parquet::Type::FIXED_LEN_BYTE_ARRAY:
    llvm_unreachable("FIXED_LEN_BYTE_ARRAY not supported");
  case ::parquet::Type::UNDEFINED:
    llvm_unreachable("UNDEFINED not supported");
  }
}

void addToCatalog(mlir::MLIRContext *ctx, Catalog &catalog,
                  const std::filesystem::path &path,
                  const ::parquet::SchemaDescriptor &schema) {
  // get the filename with the extension removed
  auto name = path.stem().string();
  auto tableAttr = TableAttr::get(ctx, name, path.string());
  catalog.addTable(tableAttr);
  for (auto i : llvm::seq(schema.num_columns())) {
    auto column = schema.Column(i);
    // TODO: Attach logical type to the column
    /*
    llvm::errs() << "logical type: " << column->logical_type()->ToString()
                 << "\n";
    */
    auto columnAttr =
        TableColumnAttr::get(ctx, tableAttr, column->name(),
                             convertPhysicalType(ctx, column->physical_type()));
    catalog.addColumn(columnAttr);
  }
}

} // namespace columnar::parquet
