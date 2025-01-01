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

class ExtractCommonSel {
private:
  mlir::Value sel;

public:
  mlir::Value input(mlir::Value v);
  mlir::Value getCommonSelectionVector();
};

} // namespace

mlir::Value ExtractCommonSel::input(mlir::Value v) {
  if (auto applyOp = v.getDefiningOp<SelApplyOp>()) {
    if (!sel || sel == applyOp.getSel()) {
      // Compatible selection vectors.
      sel = applyOp.getSel();
      return applyOp.getInput();
    } else {
      // Incompatible
      sel = nullptr;
      return v;
    }
  } else if (v.getDefiningOp<ConstantOp>()) {
    // Any selection vector is OK.
    return v;
  } else {
    // No selection vector found.
    sel = nullptr;
    return v;
  }
}

mlir::Value ExtractCommonSel::getCommonSelectionVector() { return sel; }

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

static bool readWithIdentitySel(mlir::Value v) {
  auto *op = v.getDefiningOp();
  return op && llvm::isa<ConstantOp, ReadTableOp>(op);
}

// If all inputs can be read using an identity selection vector, use that.
static mlir::LogicalResult selAddIdent(SelAddOp op,
                                       mlir::PatternRewriter &rewriter) {
  if (!llvm::all_of(op.getInputs(), readWithIdentitySel)) {
    return mlir::failure();
  }

  auto idSel =
      rewriter.create<ConstantOp>(op.getLoc(), rewriter.getAttr<SelIdAttr>());
  rewriter.replaceAllUsesWith(op.getSel(), idSel);
  rewriter.replaceAllUsesWith(op.getResults(), op.getInputs());
  return mlir::success();
}

static mlir::LogicalResult applyCmp(CmpOp op, mlir::PatternRewriter &rewriter) {
  if (op.getSel()) {
    return mlir::failure();
  }

  ExtractCommonSel extract;
  auto lhs = extract.input(op.getLhs());
  auto rhs = extract.input(op.getRhs());
  auto sel = extract.getCommonSelectionVector();
  if (!sel) {
    return mlir::failure();
  }

  auto newOp = rewriter.create<CmpOp>(op.getLoc(), op.getPred(), lhs, rhs, sel);
  rewriter.replaceOpWithNewOp<SelApplyOp>(op, newOp, sel);
  return mlir::success();
}

template <typename T>
static mlir::LogicalResult
applyNumericalBinOp(T op, mlir::PatternRewriter &rewriter) {
  if (op.getSel()) {
    return mlir::failure();
  }

  ExtractCommonSel extract;
  auto lhs = extract.input(op.getLhs());
  auto rhs = extract.input(op.getRhs());
  auto sel = extract.getCommonSelectionVector();
  if (!sel) {
    return mlir::failure();
  }

  auto newOp = rewriter.create<T>(op.getLoc(), lhs, rhs, sel);
  rewriter.replaceOpWithNewOp<SelApplyOp>(op, newOp, sel);
  return mlir::success();
}

// Whether the value represents the identity selection vector.
static bool isIdentitySel(mlir::Value v) {
  return mlir::matchPattern(v, mlir::m_Constant<SelIdAttr>(nullptr));
}

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

void AddSelectionVectors::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(lowerSelect);
  patterns.add(selAddIdent);
  patterns.add(applyCmp);
  patterns.add(applyFilter);
  patterns.add(applyNumericalBinOp<AddOp>);
  patterns.add(applyNumericalBinOp<SubOp>);
  patterns.add(applyNumericalBinOp<MulOp>);
  patterns.add(applyNumericalBinOp<DivOp>);

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