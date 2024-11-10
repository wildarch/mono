#include <columnar/Columnar.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<columnar::ColumnarDialect>();

  mlir::registerCanonicalizer();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Columnar MLIR opt\n", registry));
}
