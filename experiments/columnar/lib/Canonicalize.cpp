#include "columnar/Columnar.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace columnar {

// ============================================================================
// ========================= DROP UNUSED COLUMNS ==============================
// ============================================================================

static void setIfUsed(mlir::ValueRange values, llvm::BitVector &used) {
  for (auto [i, r] : llvm::enumerate(values)) {
    if (!r.use_empty()) {
      used.set(i);
    }
  }
}

static void addIfUsed(mlir::ValueRange values, const llvm::BitVector &used,
                      llvm::SmallVectorImpl<mlir::Value> &out) {
  for (auto [i, value] : llvm::enumerate(values)) {
    if (used[i]) {
      out.push_back(value);
    }
  }
}

static mlir::LogicalResult dropUnusedJoin(JoinOp op,
                                          mlir::PatternRewriter &rewriter) {
  llvm::BitVector usedLhs(op.getLhs().size());
  setIfUsed(op.getLhsResults(), usedLhs);

  llvm::BitVector usedRhs(op.getRhs().size());
  setIfUsed(op.getRhsResults(), usedRhs);

  // Do not allow zero columns on of the two sides
  if (usedLhs.none()) {
    usedLhs.set(0);
  }

  if (usedRhs.none()) {
    usedRhs.set(0);
  }

  if (usedLhs.all() && usedRhs.all()) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> newLhs;
  addIfUsed(op.getLhs(), usedLhs, newLhs);

  llvm::SmallVector<mlir::Value> newRhs;
  addIfUsed(op.getRhs(), usedRhs, newRhs);

  auto newOp = rewriter.create<JoinOp>(op.getLoc(), newLhs, newRhs);

  // Replace results with the new op.
  llvm::SmallVector<mlir::Value> replacedLhs;
  addIfUsed(op.getLhsResults(), usedLhs, replacedLhs);
  rewriter.replaceAllUsesWith(replacedLhs, newOp.getLhsResults());

  llvm::SmallVector<mlir::Value> replacedRhs;
  addIfUsed(op.getRhsResults(), usedRhs, replacedRhs);
  rewriter.replaceAllUsesWith(replacedRhs, newOp.getRhsResults());

  rewriter.eraseOp(op);
  return mlir::success();
}

static mlir::LogicalResult dropUnusedSelect(SelectOp op,
                                            mlir::PatternRewriter &rewriter) {
  llvm::BitVector usedInput(op->getNumOperands());
  setIfUsed(op->getResults(), usedInput);
  setIfUsed(op.getPredicates().getArguments(), usedInput);

  if (usedInput.all()) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> newInputs;
  addIfUsed(op.getInputs(), usedInput, newInputs);

  auto newOp = rewriter.create<SelectOp>(op.getLoc(), newInputs);
  rewriter.cloneRegionBefore(op.getPredicates(),
                             &newOp.getPredicates().front());
  llvm::BitVector erase = usedInput;
  erase.flip();
  newOp.getPredicates().front().eraseArguments(erase);

  llvm::SmallVector<mlir::Value> replacedResults;
  addIfUsed(op->getResults(), usedInput, replacedResults);
  rewriter.replaceAllUsesWith(replacedResults, newOp->getResults());
  return mlir::success();
}

static mlir::LogicalResult dropUnusedSubQuery(SubQueryOp op,
                                              mlir::PatternRewriter &rewriter) {
  // Remove inputs if we do not use the block arguments.
  llvm::BitVector used(op->getNumOperands());
  setIfUsed(op.getBody().getArguments(), used);
  if (used.all()) {
    return mlir::failure();
  }

  // Keep only the inputs that we do need (because the corresponding argument is
  // used).
  llvm::SmallVector<mlir::Value> newInputs;
  addIfUsed(op.getInputs(), used, newInputs);
  rewriter.modifyOpInPlace(op,
                           [&]() { op.getInputsMutable().assign(newInputs); });

  // Erase arguments we don't use.
  llvm::BitVector erase = used;
  erase.flip();
  op.getBody().front().eraseArguments(erase);

  return mlir::success();
}

// ============================================================================
// ======================= GENERAL CANONICALIZATION ===========================
// ============================================================================

static mlir::Value cloneBlockUpTo(mlir::Block *targetBlock,
                                  mlir::Value targetValue,
                                  mlir::PatternRewriter &rewriter) {
  auto sourceBlock = targetValue.getParentBlock();
  mlir::IRMapping mapping;
  mapping.map(sourceBlock->getArguments(), targetBlock->getArguments());

  for (auto &op : *sourceBlock) {
    auto *newOp = rewriter.clone(op, mapping);
    mapping.map(&op, newOp);

    if (auto newValue = mapping.lookupOrNull(targetValue)) {
      return newValue;
    }
  }

  llvm_unreachable("target value did not become available");
}

static void clonePredicate(PredicateOp op, mlir::Value cond,
                           mlir::PatternRewriter &rewriter) {
  auto newOp = rewriter.create<PredicateOp>(op->getLoc(), op.getInputs());
  auto &body = op.getBody();
  auto &newBody = newOp.getBody().emplaceBlock();
  for (auto arg : body.getArguments()) {
    newBody.addArgument(arg.getType(), arg.getLoc());
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&newBody);
  auto newCond = cloneBlockUpTo(&newBody, cond, rewriter);
  rewriter.create<PredicateEvalOp>(op.getLoc(), newCond);
}

static mlir::LogicalResult splitPredicate(PredicateOp predOp,
                                          mlir::PatternRewriter &rewriter) {
  auto evalOp =
      llvm::cast<PredicateEvalOp>(predOp.getBody().front().getTerminator());
  auto andOp = evalOp.getCond().getDefiningOp<AndOp>();
  if (!andOp) {
    return mlir::failure();
  }

  rewriter.setInsertionPointAfter(predOp);
  for (auto value : andOp.getInputs()) {
    clonePredicate(predOp, value, rewriter);
  }

  rewriter.eraseOp(predOp);
  return mlir::success();
}

// split up predicates that end with `AndOp`
static mlir::LogicalResult splitPredicates(SelectOp op,
                                           mlir::PatternRewriter &rewriter) {
  for (auto pred : op.getPredicates().getOps<PredicateOp>()) {
    if (mlir::succeeded(splitPredicate(pred, rewriter))) {
      return mlir::success();
    }
  }

  return mlir::failure();
}

void SelectOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx) {
  patterns.add(dropUnusedSelect);
  patterns.add(splitPredicates);
}

void JoinOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add(dropUnusedJoin);
}

void SubQueryOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                             mlir::MLIRContext *ctx) {
  patterns.add(dropUnusedSubQuery);
}

} // namespace columnar