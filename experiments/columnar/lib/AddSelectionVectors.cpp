#include "columnar/Columnar.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace columnar {

#define GEN_PASS_DEF_ADDSELECTIONVECTORS
#include "columnar/Passes.h.inc"

namespace {

class AddSelectionVectors
    : public impl::AddSelectionVectorsBase<AddSelectionVectors> {
public:
  using impl::AddSelectionVectorsBase<
      AddSelectionVectors>::AddSelectionVectorsBase;

  void runOnOperation() final;
};

} // namespace

/*
Base reads:
- ReadTableOp: One selection vector shared by all column reads for a given
table.

- ConstantOp: All indices 0, length depends on other inputs.

- Projections: copy from the input.

Select:
- SelectOp: Break down into filter op

Pipeline breakers:
- AggregateOp: Outputs identity selection vector. NOTE: need to add selection
vector on input side.
- HashJoinOp: Outputs identity selection vector. NOTE: need to add selection
vectors on input sides.
- UnionOp: Propagate selection vector as a regular column.
- OrderByOp: Outputs identity selection vector.
- LimitOp: Caps only the selection vector.
*/

static mlir::Value clonePredicate(PredicateOp op, mlir::ValueRange inputs,
                                  mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> predInputs;
  for (auto in : op.getInputs()) {
    auto arg = llvm::cast<mlir::BlockArgument>(in);
    predInputs.push_back(inputs[arg.getArgNumber()]);
  }

  mlir::IRMapping mapping;
  mapping.map(op.getBody().getArguments(), predInputs);

  for (auto &op : op.getBody().front()) {
    if (auto evalOp = llvm::dyn_cast<PredicateEvalOp>(op)) {
      return mapping.lookup(evalOp.getCond());
    }

    auto *newOp = rewriter.clone(op, mapping);
    mapping.map(&op, newOp);
  }

  llvm_unreachable("Missing eval op");
}

static llvm::SmallVector<mlir::Value>
applySel(mlir::ValueRange inputs, mlir::Value sel,
         mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> newInputs;
  for (auto input : inputs) {
    newInputs.push_back(
        rewriter.create<SelApplyOp>(input.getLoc(), input, sel));
  }

  return newInputs;
}

static mlir::LogicalResult lowerSelect(SelectOp op,
                                       mlir::PatternRewriter &rewriter) {
  auto inputOp = rewriter.create<SelAddOp>(op.getLoc(), op.getInputs());
  auto sel = inputOp.getSel();

  auto inputs = applySel(op.getInputs(), sel, rewriter);

  for (auto pred : op.getPredicates().getOps<PredicateOp>()) {
    llvm::SmallVector<mlir::Value> predInputs;
    for (auto in : pred.getInputs()) {
      auto arg = llvm::cast<mlir::BlockArgument>(in);
      assert(arg.getOwner() == &op.getPredicates().front());
      predInputs.push_back(inputs[arg.getArgNumber()]);
    }

    auto filter = clonePredicate(pred, inputs, rewriter);
    auto filterSel =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getAttr<SelIdAttr>());
    auto filterOp =
        rewriter.create<SelFilterOp>(op.getLoc(), sel, filter, filterSel);
    sel = filterOp.getOutSel();

    inputs = applySel(op.getInputs(), sel, rewriter);
  }

  rewriter.replaceOp(op, inputs);
  return mlir::success();
}

static mlir::LogicalResult selTable(SelAddOp op,
                                    mlir::PatternRewriter &rewriter) {
  // Verify that all inputs use the selection vector for one table.
  TableAttr common;
  for (auto input : op.getInputs()) {
    if (input.getDefiningOp<ConstantOp>()) {
      continue;
    } else if (auto readOp = input.getDefiningOp<ReadColumnOp>()) {
      auto table = readOp.getColumn().getTable();
      if (!common) {
        common = table;
      } else if (common != table) {
        return mlir::failure();
      }
    }
  }

  if (!common) {
    return mlir::failure();
  }

  auto tableSel = rewriter.create<SelTableOp>(op.getLoc(), common);
  rewriter.replaceAllUsesWith(op.getSel(), tableSel);
  rewriter.replaceAllUsesWith(op.getResults(), op.getInputs());
  return mlir::success();
}

static mlir::LogicalResult addCmp(CmpOp op, mlir::PatternRewriter &rewriter) {
  if (op.getSel()) {
    return mlir::failure();
  }

  auto selAddOp = rewriter.create<SelAddOp>(op.getLoc(), op.getOperands());
  auto lhs = selAddOp->getResult(0);
  auto rhs = selAddOp->getResult(1);
  auto sel = selAddOp.getSel();

  auto newOp = rewriter.create<CmpOp>(op.getLoc(), op.getPred(), lhs, rhs, sel);
  rewriter.replaceOpWithNewOp<SelApplyOp>(op, newOp, sel);
  return mlir::success();
}

// Whether the value represents the identity selection vector.
static bool isIdentitySel(mlir::Value v) {
  return mlir::matchPattern(v, mlir::m_Constant<SelIdAttr>(nullptr));
}

template <typename T>
static mlir::LogicalResult addNumericalBinOp(T op,
                                             mlir::PatternRewriter &rewriter) {
  if (op.getSel()) {
    return mlir::failure();
  }

  auto selAddOp = rewriter.create<SelAddOp>(op.getLoc(), op.getOperands());
  auto lhs = selAddOp->getResult(0);
  auto rhs = selAddOp->getResult(1);
  auto sel = selAddOp.getSel();
  auto newOp = rewriter.create<T>(op.getLoc(), lhs, rhs, sel);
  rewriter.replaceOpWithNewOp<SelApplyOp>(op, newOp, sel);
  return mlir::success();
}

static mlir::LogicalResult addQueryOutput(QueryOutputOp op,
                                          mlir::PatternRewriter &rewriter) {
  if (op.getSel()) {
    return mlir::failure();
  }

  auto selAddOp = rewriter.create<SelAddOp>(op.getLoc(), op.getColumns());
  rewriter.modifyOpInPlace(op, [&]() {
    op.getColumnsMutable().assign(selAddOp.getResults());
    op.getSelMutable().assign(selAddOp.getSel());
  });

  return mlir::success();
}

// Fold selection vector into input.
static mlir::LogicalResult applyFilter(SelFilterOp op,
                                       mlir::PatternRewriter &rewriter) {
  auto applyOp = op.getFilter().getDefiningOp<SelApplyOp>();
  if (!applyOp || !isIdentitySel(op.getFilterSel())) {
    return mlir::failure();
  }

  rewriter.modifyOpInPlace(op, [&]() {
    op.getFilterMutable().assign(applyOp.getInput());
    op.getFilterSelMutable().assign(applyOp.getSel());
  });
  return mlir::success();
}

// If all inputs are SelApplyOp with the same selection vector, the ops cancel
// out.
static mlir::LogicalResult addApply(SelAddOp op,
                                    mlir::PatternRewriter &rewriter) {
  mlir::Value sel;
  for (auto input : op.getInputs()) {
    if (input.getDefiningOp<ConstantOp>()) {
      // Compatible with any selection vector.
      continue;
    }

    auto applyOp = input.getDefiningOp<SelApplyOp>();
    if (!applyOp) {
      // All inputs must be apply.
      return mlir::failure();
    }

    if (sel && applyOp.getSel() != sel) {
      // Incompatible selection vectors.
      return mlir::failure();
    }

    sel = applyOp.getSel();
  }

  if (!sel) {
    // All inputs are constants
    sel =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getAttr<SelIdAttr>());
  }

  llvm::SmallVector<mlir::Value> replacements;
  for (auto input : op.getInputs()) {
    if (input.getDefiningOp<ConstantOp>()) {
      replacements.push_back(input);
    } else {
      replacements.push_back(input.getDefiningOp<SelApplyOp>().getInput());
    }
  }

  replacements.push_back(sel);

  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

void AddSelectionVectors::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(lowerSelect);
  patterns.add(selTable);
  patterns.add(addCmp);
  patterns.add(addNumericalBinOp<AddOp>);
  patterns.add(addNumericalBinOp<SubOp>);
  patterns.add(addNumericalBinOp<MulOp>);
  patterns.add(addNumericalBinOp<DivOp>);
  patterns.add(addQueryOutput);

  patterns.add(applyFilter);
  patterns.add(addApply);

  if (mlir ::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                       std::move(patterns)))) {
    return signalPassFailure();
  }

  getOperation()->walk([](SelAddOp op) {
    auto diag = op->emitOpError("remains after rewrites");
    for (auto *user : op->getUsers()) {
      diag.attachNote(user->getLoc()) << "used by " << user;
    }
  });

  getOperation()->walk([](SelApplyOp op) {
    auto diag = op->emitWarning("op 'sel.apply' remains after rewrites");
    for (auto *user : op->getUsers()) {
      diag.attachNote(user->getLoc()) << "used by " << user;
    }
  });
}

} // namespace columnar