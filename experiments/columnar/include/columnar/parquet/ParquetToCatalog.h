#pragma once

#include <filesystem>

#include <mlir/IR/MLIRContext.h>

#include <parquet/metadata.h>

#include "columnar/Catalog.h"

namespace columnar::parquet {

void addToCatalog(mlir::MLIRContext *ctx, Catalog &catalog,
                  const std::filesystem::path &path,
                  const ::parquet::SchemaDescriptor &schema);

} // namespace columnar::parquet
