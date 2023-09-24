#include <llvm/ADT/TypeSwitch.h>

#include "execution/Batch.h"
#include "execution/expression/BinaryOperatorExpression.h"
#include "execution/expression/ColumnExpression.h"
#include "execution/expression/ConstantExpression.h"
#include "execution/expression/ExpressionEvaluator.h"
#include "execution/expression/ExpressionVisitor.h"
#include "execution/expression/IR/ExpressionOps.h"
#include "execution/operator/IR/OperatorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace execution {

ConstantValue applyBinOp(ConstantValue lhs, BinaryOperator op,
                         ConstantValue rhs) {
  switch (op) {
  case BinaryOperator::LE_INT32:
    return std::get<int32_t>(lhs) <= std::get<int32_t>(rhs);
  }

  throw std::logic_error("invalid enum");
}

ConstantValue evaluate(const Expression &expr, const Batch &batch,
                       uint32_t row) {
  return ExpressionVisitor{
      [&](const BinaryOperatorExpression &expr) {
        return applyBinOp(evaluate(*expr.lhs(), batch, row), expr.op(),
                          evaluate(*expr.rhs(), batch, row));
      },
      [&](const ColumnExpression &expr) -> ConstantValue {
        switch (expr.type()) {
#define CASE(t)                                                                \
  case PhysicalColumnType::t:                                                  \
    return batch.columns()                                                     \
        .at(expr.idx().value())                                                \
        .get<PhysicalColumnType::t>()[row];                                    \
    break;
          CASE(INT32)
          CASE(DOUBLE)
          CASE(STRING_PTR)
#undef CASE
        }
        throw std::logic_error("invalid enum");
      },
      [](const ConstantExpression &expr) { return expr.value(); },
  }
      .visit(expr);
}

static ConstantValue evaluate(mlir::Value val, const Batch &batch,
                              uint32_t row) {
  // TODO: CSE
  if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
    if (arg.getType().isInteger(/*width=*/32)) {
      return batch.columns()
          .at(arg.getArgNumber())
          .get<PhysicalColumnType::INT32>()[row];
    }

    arg.dump();
    throw std::logic_error("type not supported");
  }
  return evaluate(val.getDefiningOp(), batch, row);
}

ConstantValue evaluate(mlir::Operation *op, const Batch &batch, uint32_t row) {
  using namespace qoperator;
  return llvm::TypeSwitch<mlir::Operation *, ConstantValue>(op)
      .Case<mlir::arith::ConstantIntOp>([&](mlir::arith::ConstantIntOp op) {
        return std::int32_t(
            llvm::cast<mlir::IntegerAttr>(op.getValue()).getInt());
      })
      .Case<mlir::arith::CmpIOp>([&](mlir::arith::CmpIOp op) {
        auto lhs = std::get<int32_t>(evaluate(op.getLhs(), batch, row));
        auto rhs = std::get<int32_t>(evaluate(op.getRhs(), batch, row));
        switch (op.getPredicate()) {
        case mlir::arith::CmpIPredicate::sle:
          return lhs <= rhs;
        default:
          op->emitOpError("execution not supported");
          throw std::logic_error("execution not supported");
        }
      })
      .Case<FilterReturnOp>([&](FilterReturnOp op) {
        return evaluate(op.getCondition(), batch, row);
      })
      .Default([](mlir::Operation *op) -> ConstantValue {
        op->emitOpError("execution not supported");
        throw std::logic_error("execution not supported");
      });
}

} // namespace execution