#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

#include "columnar/Columnar.h"

namespace columnar {

namespace {

class ColumnarDialectLLVMIRTranslationInterface
    : public mlir::LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  mlir::LogicalResult convertOperation(
      mlir::Operation *op, llvm::IRBuilderBase &builder,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const override {
    return mlir::success(llvm::isa<PipelineRefOp>(op));
  }
};

} // namespace

void registerLLVMTranslation(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, ColumnarDialect *dialect) {
    dialect->addInterfaces<ColumnarDialectLLVMIRTranslationInterface>();
  });
}

} // namespace columnar