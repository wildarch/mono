#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <columnar/Columnar.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<columnar::ColumnarDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Bufferization
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::registerOneShotBufferize();
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::registerConvertLinalgToLoopsPass();
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::registerCanonicalizer();
  columnar::registerPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Columnar MLIR opt\n", registry));
}
