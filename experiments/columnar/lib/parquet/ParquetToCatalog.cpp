#include <filesystem>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

#include <mlir/IR/MLIRContext.h>
#include <parquet/metadata.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include "columnar/Catalog.h"
#include "columnar/Columnar.h"

namespace columnar::parquet {

static mlir::Type convertPhysicalType(mlir::MLIRContext *ctx,
                                      ::parquet::Type::type type) {
  switch (type) {
  case ::parquet::Type::BOOLEAN:
    return mlir::IntegerType::get(ctx, 1);
  case ::parquet::Type::INT32:
    return mlir::IntegerType::get(ctx, 32);
  case ::parquet::Type::INT64:
    return mlir::IntegerType::get(ctx, 64);
  case ::parquet::Type::FLOAT:
    return mlir::Float32Type::get(ctx);
  case ::parquet::Type::DOUBLE:
    return mlir::Float64Type::get(ctx);
  case ::parquet::Type::INT96:
    llvm_unreachable("INT96 not supported");
  case ::parquet::Type::BYTE_ARRAY:
    return columnar::ByteArrayType::get(ctx);
  case ::parquet::Type::FIXED_LEN_BYTE_ARRAY:
    llvm_unreachable("FIXED_LEN_BYTE_ARRAY not supported");
  case ::parquet::Type::UNDEFINED:
    llvm_unreachable("UNDEFINED not supported");
  }
}

static mlir::Type convertLogicalType(mlir::MLIRContext *ctx,
                                     const ::parquet::LogicalType &type) {
  if (type.is_int()) {
    const auto &intType = static_cast<const ::parquet::IntLogicalType &>(type);
    return mlir::IntegerType::get(ctx, intType.bit_width(),
                                  intType.is_signed()
                                      ? mlir::IntegerType::Signed
                                      : mlir::IntegerType::Unsigned);
  } else if (type.is_string()) {
    return StringType::get(ctx);
  } else if (type.is_decimal()) {
    const auto &decType =
        static_cast<const ::parquet::DecimalLogicalType &>(type);
    // TODO: Add precision and scale
    return DecimalType::get(ctx);
  } else if (type.is_date()) {
    return DateType::get(ctx);
  }

  llvm::errs() << "Cannot convert logical type: " << type.ToString() << "\n";
  llvm_unreachable("Cannot convert logical type");
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
    auto columnAttr =
        TableColumnAttr::get(ctx, tableAttr, i, column->name(),
                             convertLogicalType(ctx, *column->logical_type()),
                             convertPhysicalType(ctx, column->physical_type()));
    catalog.addColumn(columnAttr);
  }
}

} // namespace columnar::parquet
