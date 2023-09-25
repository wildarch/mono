#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Operation.h>

#include "execution/expression/bytecode/ExpressionByteCode.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace execution {
namespace expression {
namespace bytecode {

std::uint16_t RegisterAllocator::analyseRegistersNeeded(mlir::Value val) {
  if (llvm::isa<mlir::BlockArgument>(val)) {
    // Value must be loaded from memory into a register.
    _registersNeeded[val] = 1;
    return 1;
  }

  auto op = val.getDefiningOp();
  assert(op != nullptr);

  if (llvm::isa<mlir::arith::ConstantOp>(op)) {
    // Value must be loaded into a register.
    _registersNeeded[val] = 1;
    return 1;
  }

  assert(op->getNumOperands() == 2);
  auto lhs = analyseRegistersNeeded(op->getOperand(0));
  auto rhs = analyseRegistersNeeded(op->getOperand(1));

  reg_t needed;
  if (lhs == rhs) {
    // We can execute one of the subtrees (arbitrary) first, store that in a
    // register, then compute the other one.
    needed = lhs + 1;
  } else {
    // Compute the expensive one first. The cheaper one should use at least one
    // fewer register, which we use to store the intermediate result.
    needed = std::max(lhs, rhs);
  }
  _registersNeeded[val] = needed;
  return needed;
}

reg_t ByteCodeCompiler::compile(mlir::Value val, reg_t nextFreeReg) {
  if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
    assert(arg.getType().isInteger(/*width=*/32));
    _code.emplace_back(Instruction{
        .op = Op::LOAD_I32,
        .rd = nextFreeReg++,
        .col_idx = static_cast<uint16_t>(arg.getArgNumber()),
    });
  }

  return llvm::TypeSwitch<mlir::Operation *, reg_t>(val.getDefiningOp())
      .Default([](mlir::Operation *val) -> reg_t {
        val->dump();
        llvm_unreachable("not supported");
      });
}

std::vector<Instruction> ByteCodeCompiler::compile(mlir::Value val) {
  // TODO
  ByteCodeCompiler compiler;
  compiler.compile(val, /*nextFreeReg=*/0);
  return std::move(compiler._code);
}

} // namespace bytecode
} // namespace expression
} // namespace execution