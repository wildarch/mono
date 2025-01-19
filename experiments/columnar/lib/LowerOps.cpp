#include "columnar/Columnar.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace columnar {

#define GEN_PASS_DEF_LOWEROPS
#include "columnar/Passes.h.inc"

namespace {

class LowerOps : public impl::LowerOpsBase<LowerOps> {
public:
  using impl::LowerOpsBase<LowerOps>::LowerOpsBase;

  void runOnOperation() final;
};

struct RuntimeFuncOps {
  mlir::func::FuncOp selScannerNewOp;
  mlir::func::FuncOp tableColumnScannerNewOp;
};

class ExternalFuncBuilder {
private:
  mlir::IRRewriter _rewriter;

public:
  ExternalFuncBuilder(mlir::ModuleOp module) : _rewriter(module.getContext()) {
    _rewriter.setInsertionPointToStart(module.getBody());
  }

  mlir::Type getTableIdType() { return _rewriter.getI64Type(); }

  mlir::Type getColumnIdType() { return _rewriter.getI32Type(); }

  mlir::Type getPointerType() { return _rewriter.getIndexType(); }

  mlir::func::FuncOp build(llvm::StringRef name, mlir::TypeRange inputs,
                           mlir::TypeRange results);
};

} // namespace

mlir::func::FuncOp ExternalFuncBuilder::build(llvm::StringRef name,
                                              mlir::TypeRange inputs,
                                              mlir::TypeRange results) {
  return _rewriter.create<mlir::func::FuncOp>(
      _rewriter.getUnknownLoc(), _rewriter.getStringAttr(name),
      _rewriter.getFunctionType(inputs, results),
      _rewriter.getStringAttr("private"),
      /*arg_attrs=*/nullptr,
      /*res_attrs=*/nullptr);
}

void LowerOps::runOnOperation() {
  // Register external functions
  ExternalFuncBuilder funcBuilder(getOperation());
  RuntimeFuncOps runtimeFuncs;
  runtimeFuncs.selScannerNewOp =
      funcBuilder.build("sel_scanner_new", funcBuilder.getTableIdType(),
                        funcBuilder.getPointerType());
  runtimeFuncs.tableColumnScannerNewOp =
      funcBuilder.build("table_columnar_scanner_new",
                        mlir::TypeRange{funcBuilder.getTableIdType(),
                                        funcBuilder.getColumnIdType()},
                        funcBuilder.getPointerType());
}

} // namespace columnar