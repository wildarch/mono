#include "columnar/Columnar.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace columnar {

#define GEN_PASS_DEF_LOWERTOLLVM
#include "columnar/Passes.h.inc"

namespace {

class LowerToLLVM : public impl::LowerToLLVMBase<LowerToLLVM> {
public:
  using impl::LowerToLLVMBase<LowerToLLVM>::LowerToLLVMBase;

  void runOnOperation() final;
};

} // namespace

static mlir::LogicalResult findRuntimeCallsIn(
    mlir::Operation *op,
    llvm::DenseMap<mlir::StringAttr, mlir::FunctionType> &called) {
  bool hadError = false;
  op->walk([&](RuntimeCallOp op) {
    auto type = mlir::FunctionType::get(
        op->getContext(), op.getInputs().getTypes(), op->getResultTypes());
    auto exist = called.lookup(op.getFuncAttr());
    if (!exist) {
      called[op.getFuncAttr()] = type;
    } else if (type != exist) {
      op->emitError("runtime function '")
          << op.getFunc() << "' is used with different function types ("
          << exist << " vs. " << type << ")";
      hadError = true;
    }
  });

  return mlir::failure(hadError);
}

void LowerToLLVM::runOnOperation() {
  // Find called runtime functions
  llvm::DenseMap<mlir::StringAttr, mlir::FunctionType> called;
  if (mlir::failed(findRuntimeCallsIn(getOperation(), called))) {
    return signalPassFailure();
  }

  // Register as external functions
  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(getOperation().getBody());
  auto funcPrivateAttr = builder.getStringAttr("private");
  for (auto [name, type] : called) {
    builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), name, type, funcPrivateAttr,
        /*arg_attrs=*/nullptr, /*res_attrs*/ nullptr);
  }

  // Replace with function calls
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<RuntimeCallOp> callOps;
  getOperation()->walk([&](RuntimeCallOp op) { callOps.push_back(op); });

  for (auto op : callOps) {
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, op.getFuncAttr(), op.getResultTypes(), op.getInputs());
  }

  // TODO: Group pipeline arguments into structs.
  // TODO: Replace pipeline blocks with references to functions.

  // TODO: Invoke LLVM lowering to lower the functions.
}

} // namespace columnar