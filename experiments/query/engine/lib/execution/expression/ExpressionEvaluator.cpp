#include <llvm-17/llvm/Support/ErrorHandling.h>
#include <llvm/ADT/TypeSwitch.h>

#include "execution/Batch.h"
#include "execution/expression/ExpressionEvaluator.h"
#include "execution/expression/IR/ExpressionOps.h"
#include "execution/operator/IR/OperatorOps.h"
#include "execution/operator/IR/OperatorTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace execution {

AnyValue evaluate(mlir::Value val, const Batch &batch, uint32_t row) {
  // TODO: CSE
  if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
    if (arg.getType().isInteger(/*width=*/32)) {
      return batch.columns()
          .at(arg.getArgNumber())
          .get<PhysicalColumnType::INT32>()[row];
    }
    if (arg.getType().isInteger(/*width=*/64)) {
      return batch.columns()
          .at(arg.getArgNumber())
          .get<PhysicalColumnType::INT64>()[row];
    }
    if (arg.getType().isa<qoperator::VarcharType>()) {
      return batch.columns()
          .at(arg.getArgNumber())
          .get<PhysicalColumnType::STRING>()[row];
    }

    arg.dump();
    llvm_unreachable("type not supported");
  }
  return evaluate(val.getDefiningOp(), batch, row);
}

template <typename T> static T addi(AnyValue lhs, AnyValue rhs) {
  return std::get<T>(lhs) + std::get<T>(rhs);
}

template <typename T> static T subi(AnyValue lhs, AnyValue rhs) {
  return std::get<T>(lhs) - std::get<T>(rhs);
}

template <typename T> static T muli(AnyValue lhs, AnyValue rhs) {
  return std::get<T>(lhs) * std::get<T>(rhs);
}

template <typename T> static T divsi(AnyValue lhs, AnyValue rhs) {
  return std::get<T>(lhs) / std::get<T>(rhs);
}

AnyValue evaluate(mlir::Operation *op, const Batch &batch, uint32_t row) {
  using namespace qoperator;
  return llvm::TypeSwitch<mlir::Operation *, AnyValue>(op)
      .Case<mlir::arith::ConstantIntOp>(
          [&](mlir::arith::ConstantIntOp op) -> AnyValue {
            int64_t val = llvm::cast<mlir::IntegerAttr>(op.getValue()).getInt();
            if (op.getType().isInteger(/*width=*/32)) {
              return std::int32_t(val);
            } else if (op.getType().isInteger(/*width=*/64)) {
              return std::int64_t(val);
            } else if (op.getType().isInteger(/*width*/ 1)) {
              return bool(val);
            } else {
              op->emitOpError("execution not supported");
              llvm_unreachable("execution not supported");
            }
          })
      .Case<mlir::arith::AddIOp>([&](mlir::arith::AddIOp op) -> AnyValue {
        auto lhs = evaluate(op.getLhs(), batch, row);
        auto rhs = evaluate(op.getRhs(), batch, row);
        if (op.getType().isInteger(/*width=*/32)) {
          return addi<int32_t>(lhs, rhs);
        } else if (op.getType().isInteger(/*width=*/64)) {
          return addi<int64_t>(lhs, rhs);
        } else {
          op->emitOpError("execution not supported");
          llvm_unreachable("execution not supported");
        }
      })
      .Case<mlir::arith::SubIOp>([&](mlir::arith::SubIOp op) -> AnyValue {
        auto lhs = evaluate(op.getLhs(), batch, row);
        auto rhs = evaluate(op.getRhs(), batch, row);
        if (op.getType().isInteger(/*width=*/32)) {
          return subi<int32_t>(lhs, rhs);
        } else if (op.getType().isInteger(/*width=*/64)) {
          return subi<int64_t>(lhs, rhs);
        } else {
          op->emitOpError("execution not supported");
          llvm_unreachable("execution not supported");
        }
      })
      .Case<mlir::arith::MulIOp>([&](mlir::arith::MulIOp op) -> AnyValue {
        auto lhs = evaluate(op.getLhs(), batch, row);
        auto rhs = evaluate(op.getRhs(), batch, row);
        if (op.getType().isInteger(/*width=*/32)) {
          return muli<int32_t>(lhs, rhs);
        } else if (op.getType().isInteger(/*width=*/64)) {
          return muli<int64_t>(lhs, rhs);
        } else {
          op->emitOpError("execution not supported");
          llvm_unreachable("execution not supported");
        }
      })
      .Case<mlir::arith::DivSIOp>([&](mlir::arith::DivSIOp op) -> AnyValue {
        auto lhs = evaluate(op.getLhs(), batch, row);
        auto rhs = evaluate(op.getRhs(), batch, row);
        if (op.getType().isInteger(/*width=*/32)) {
          return divsi<int32_t>(lhs, rhs);
        } else if (op.getType().isInteger(/*width=*/64)) {
          return divsi<int64_t>(lhs, rhs);
        } else {
          op->emitOpError("execution not supported");
          llvm_unreachable("execution not supported");
        }
      })
      .Case<mlir::arith::CmpIOp>([&](mlir::arith::CmpIOp op) {
        auto lhs = std::get<int32_t>(evaluate(op.getLhs(), batch, row));
        auto rhs = std::get<int32_t>(evaluate(op.getRhs(), batch, row));
        switch (op.getPredicate()) {
        case mlir::arith::CmpIPredicate::sle:
          return lhs <= rhs;
        default:
          op->emitOpError("execution not supported");
          llvm_unreachable("execution not supported");
        }
      })
      .Case<FilterReturnOp>([&](FilterReturnOp op) {
        return evaluate(op.getCondition(), batch, row);
      })
      .Default([](mlir::Operation *op) -> AnyValue {
        op->emitOpError("execution not supported");
        llvm_unreachable("execution not supported");
      });
}

} // namespace execution