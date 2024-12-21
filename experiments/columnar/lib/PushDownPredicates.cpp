#include "columnar/Columnar.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace columnar {

#define GEN_PASS_DEF_PUSHDOWNPREDICATES
#include "columnar/Passes.h.inc"

namespace {

class PushDownPredicates
    : public impl::PushDownPredicatesBase<PushDownPredicates> {
public:
  using impl::PushDownPredicatesBase<
      PushDownPredicates>::PushDownPredicatesBase;

  void runOnOperation() final;
};

enum class JoinSide {
  LHS,
  RHS,
};

struct PredicateArgMapping {
  // Block argument number to operand index on the child op.
  llvm::SmallDenseMap<unsigned int, unsigned int> mapping;

  PredicateArgMapping(SelectOp selectOp, AggregateOp aggOp);
  PredicateArgMapping(SelectOp selectOp, JoinOp joinOp, JoinSide side);
  PredicateArgMapping(SelectOp selectOp, SelectOp childOp);

  bool canPushDown(PredicateOp predOp);

  void movePredicate(PredicateOp predOp, SelectOp newParent,
                     mlir::PatternRewriter &rewriter);
};

} // namespace

PredicateArgMapping::PredicateArgMapping(SelectOp selectOp, AggregateOp aggOp) {
  for (auto &oper : selectOp->getOpOperands()) {
    if (auto res = llvm::dyn_cast<mlir::OpResult>(oper.get())) {
      assert(res.getOwner() == aggOp);
      // We cannot push down predicates if they depend on aggregated values, so
      // we only map group-by columns.
      if (res.getResultNumber() < aggOp.getGroupBy().size()) {
        mapping[oper.getOperandNumber()] = res.getResultNumber();
      }
    }
  }
}

PredicateArgMapping::PredicateArgMapping(SelectOp selectOp, JoinOp joinOp,
                                         JoinSide side) {
  for (auto &oper : selectOp->getOpOperands()) {
    if (auto res = llvm::dyn_cast<mlir::OpResult>(oper.get())) {
      assert(res.getOwner() == joinOp);

      auto nLhs = joinOp.getLhs().size();
      if (side == JoinSide::LHS && res.getResultNumber() < nLhs) {
        mapping[oper.getOperandNumber()] = res.getResultNumber();
      } else if (side == JoinSide::RHS && (res.getResultNumber() >= nLhs)) {
        assert(res.getResultNumber() - nLhs < joinOp.getRhs().size());
        mapping[oper.getOperandNumber()] = res.getResultNumber() - nLhs;
      }
    }
  }
}

PredicateArgMapping::PredicateArgMapping(SelectOp selectOp, SelectOp childOp) {
  for (auto &oper : selectOp->getOpOperands()) {
    auto res = llvm::cast<mlir::OpResult>(oper.get());
    assert(res.getOwner() == childOp);
    mapping[oper.getOperandNumber()] = res.getResultNumber();
  }
}

bool PredicateArgMapping::canPushDown(PredicateOp predOp) {
  return llvm::all_of(predOp.getInputs(), [this](mlir::Value v) {
    auto arg = llvm::cast<mlir::BlockArgument>(v);
    return mapping.contains(arg.getArgNumber());
  });
}

void PredicateArgMapping::movePredicate(PredicateOp predOp,
                                        SelectOp newParentOp,
                                        mlir::PatternRewriter &rewriter) {
  auto oldParentOp = predOp.getParentOp();
  auto &oldBlock = oldParentOp.getPredicates().front();
  auto &newBlock = newParentOp.getPredicates().front();
  mlir::IRMapping argMap;
  for (auto [from, to] : mapping) {
    argMap.map(oldBlock.getArgument(from), newBlock.getArgument(to));
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&newBlock);
  rewriter.clone(*predOp, argMap);
  rewriter.eraseOp(predOp);
}

static auto isDefinedBy(mlir::Operation *op) {
  return [op](mlir::Value v) { return v.getDefiningOp() == op; };
}

static void replaceChild(SelectOp op, mlir::Operation *oldChild,
                         mlir::Operation *newChild,
                         mlir::PatternRewriter &rewriter) {
  // Build the mapping.
  mlir::IRMapping inputMapping;
  inputMapping.map(oldChild->getResults(), newChild->getResults());

  // Remap all inputs.
  llvm::SmallVector<mlir::Value> newInputs;
  for (auto v : op.getInputs()) {
    assert(inputMapping.contains(v) && "Not a result of oldChild");
    newInputs.push_back(inputMapping.lookup(v));
  }

  rewriter.modifyOpInPlace(op,
                           [&]() { op.getInputsMutable().assign(newInputs); });
}

static mlir::LogicalResult pushDownAggregate(SelectOp selectOp,
                                             AggregateOp aggOp,
                                             PredicateOp predOp,
                                             mlir::PatternRewriter &rewriter) {
  PredicateArgMapping predMap(selectOp, aggOp);
  if (!predMap.canPushDown(predOp)) {
    return mlir::failure();
  }

  // Insert the new select over the aggregation inputs.
  rewriter.setInsertionPoint(aggOp);
  auto newSelect =
      rewriter.create<SelectOp>(selectOp.getLoc(), aggOp->getOperands());

  predMap.movePredicate(predOp, newSelect, rewriter);

  // Create a new aggregation over the filtered inputs.
  mlir::IRMapping inputMapping;
  inputMapping.map(aggOp.getOperands(), newSelect->getResults());
  auto newAgg = rewriter.clone(*aggOp, inputMapping);

  // Update existing select to use the new aggregation as input.
  replaceChild(selectOp, aggOp, newAgg, rewriter);

  return mlir::success();
}

static mlir::LogicalResult pushDownAggregate(SelectOp op,
                                             mlir::PatternRewriter &rewriter) {
  // All inputs derive from one AggregateOp.
  auto aggOp = op.getInputs()[0].getDefiningOp<AggregateOp>();
  if (!aggOp || !llvm::all_of(op.getInputs(), isDefinedBy(aggOp))) {
    return mlir::failure();
  }

  // For each predicate:
  llvm::SmallVector<PredicateOp> preds(
      op.getPredicates().front().getOps<PredicateOp>());
  for (auto pred : preds) {
    // Can we move it to before the AggregateOp?
    if (mlir::succeeded(pushDownAggregate(op, aggOp, pred, rewriter))) {
      return mlir::success();
    }
  }

  return mlir::failure();
}

static mlir::LogicalResult pushDownJoinLHS(SelectOp selectOp, JoinOp joinOp,
                                           PredicateOp predOp,
                                           mlir::PatternRewriter &rewriter) {
  PredicateArgMapping predMap(selectOp, joinOp, JoinSide::LHS);
  if (!predMap.canPushDown(predOp)) {
    return mlir::failure();
  }

  // New select over the LHS inputs
  auto newSelect =
      rewriter.create<SelectOp>(selectOp.getLoc(), joinOp.getLhs());

  predMap.movePredicate(predOp, newSelect, rewriter);

  // Create new join.
  auto newJoinOp = rewriter.create<JoinOp>(
      joinOp.getLoc(), newSelect->getResults(), joinOp.getRhs());

  // Update existing select to use the new join as input.
  replaceChild(selectOp, joinOp, newJoinOp, rewriter);

  return mlir::success();
}

static mlir::LogicalResult pushDownJoinRHS(SelectOp selectOp, JoinOp joinOp,
                                           PredicateOp predOp,
                                           mlir::PatternRewriter &rewriter) {
  PredicateArgMapping predMap(selectOp, joinOp, JoinSide::RHS);
  if (!predMap.canPushDown(predOp)) {
    return mlir::failure();
  }

  // New select over the RHS inputs
  auto newSelect =
      rewriter.create<SelectOp>(selectOp.getLoc(), joinOp.getRhs());

  predMap.movePredicate(predOp, newSelect, rewriter);

  // Create new join.
  auto newJoinOp = rewriter.create<JoinOp>(joinOp.getLoc(), joinOp.getLhs(),
                                           newSelect->getResults());

  // Update existing select to use the new join as input.
  replaceChild(selectOp, joinOp, newJoinOp, rewriter);

  return mlir::success();
}

static mlir::LogicalResult pushDownJoin(SelectOp op,
                                        mlir::PatternRewriter &rewriter) {
  // All inputs derive from one JoinOp.
  auto joinOp = op.getInputs()[0].getDefiningOp<JoinOp>();
  if (!joinOp || !llvm::all_of(op.getInputs(), isDefinedBy(joinOp))) {
    return mlir::failure();
  }

  // For each predicate:
  llvm::SmallVector<PredicateOp> preds(
      op.getPredicates().front().getOps<PredicateOp>());
  for (auto pred : preds) {
    // Can we move it to one of the join children?
    if (mlir::succeeded(pushDownJoinLHS(op, joinOp, pred, rewriter))) {
      return mlir::success();
    } else if (mlir::succeeded(pushDownJoinRHS(op, joinOp, pred, rewriter))) {
      return mlir::success();
    }
  }

  return mlir::failure();
}

static mlir::LogicalResult pushDownUnion(SelectOp op,
                                         mlir::PatternRewriter &rewriter) {
  // All inputs derive from one UnionOp.
  auto unionOp = op.getInputs()[0].getDefiningOp<UnionOp>();
  if (!unionOp || !llvm::all_of(op.getInputs(), isDefinedBy(unionOp))) {
    return mlir::failure();
  }

  // Pushing down into unions is trivial: The inputs have the same columns as
  // the result and in the same order, so we can simply clone the SelectOp over
  // both sides, then union them together again.

  // Clone over LHS
  mlir::IRMapping lhsMapping;
  lhsMapping.map(unionOp->getResults(), unionOp.getLhs());
  auto lhsSelect = rewriter.clone(*op, lhsMapping);

  // Clone over RHS
  mlir::IRMapping rhsMapping;
  rhsMapping.map(unionOp->getResults(), unionOp.getRhs());
  auto rhsSelect = rewriter.clone(*op, rhsMapping);

  // Union over the select
  rewriter.replaceOpWithNewOp<UnionOp>(op, op.getResultTypes(),
                                       lhsSelect->getResults(),
                                       rhsSelect->getResults());
  return mlir::success();
}

static mlir::LogicalResult pushDownProjection(SelectOp op,
                                              unsigned int operandIdx,
                                              mlir::PatternRewriter &rewriter) {
  auto projOp = op->getOperand(operandIdx).getDefiningOp();

  llvm::SmallVector<mlir::Value> newInputs(op.getInputs());
  // Add the projection inputs to the select.
  auto projInputs = projOp->getOperands();
  newInputs.append(projInputs.begin(), projInputs.end());

  // Remove projection result
  newInputs.erase(newInputs.begin() + operandIdx);

  auto newSelect = rewriter.create<SelectOp>(op.getLoc(), newInputs);
  // NOTE: Builder adds a block, but we move it from the other op.
  newSelect.getPredicates().front().erase();
  rewriter.inlineRegionBefore(op.getPredicates(), newSelect.getPredicates(),
                              newSelect.getPredicates().end());

  // Add arguments for the operands we added
  auto &selectBody = newSelect.getPredicates().front();
  llvm::SmallVector<mlir::Value> newSelectArgs;
  for (auto input : projInputs) {
    auto arg = selectBody.addArgument(input.getType(), input.getLoc());
    newSelectArgs.push_back(arg);
  }

  auto replacedArg = selectBody.getArgument(operandIdx);
  for (auto pred : newSelect.getPredicates().front().getOps<PredicateOp>()) {
    auto replacedIt = llvm::find(pred.getInputs(), replacedArg);
    if (replacedIt == pred.getInputs().end()) {
      // No need to change this predicate
      continue;
    }

    auto replacedPredArgIdx =
        std::distance(pred.getInputs().begin(), replacedIt);

    // Add the new inputs and remove the old one.
    rewriter.modifyOpInPlace(pred, [&]() {
      auto inputs = pred.getInputsMutable();
      inputs.erase(replacedPredArgIdx, 1);
      inputs.append(newSelectArgs);
    });

    // Add the new arguments (the old one is removed later)
    auto &predBlock = pred.getBody().front();
    mlir::IRMapping projMapping;
    for (auto input : projInputs) {
      auto arg = predBlock.addArgument(input.getType(), input.getLoc());
      projMapping.map(input, arg);
    }

    // Recreate the op inside of predicate.
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&predBlock);
    auto *newProj = rewriter.clone(*projOp, projMapping);
    rewriter.replaceAllUsesWith(predBlock.getArgument(replacedPredArgIdx),
                                newProj->getResult(0));

    // Now that there are no remaining uses, delete the old arg.
    predBlock.eraseArgument(replacedPredArgIdx);
  }

  // Now that there are no remaining uses, delete the old arg.
  selectBody.eraseArgument(operandIdx);

  // Recreate the op after the select
  auto newProjInputs = newSelect->getResults().take_back(projInputs.size());
  mlir::IRMapping projMapping;
  projMapping.map(projInputs, newProjInputs);
  auto newProj = rewriter.clone(*projOp, projMapping);

  // Replace uses
  llvm::SmallVector<mlir::Value> newResults(
      newSelect->getResults().take_front(op->getNumResults() - 1));
  newResults.insert(newResults.begin() + operandIdx, newProj->getResult(0));
  rewriter.replaceOp(op, newResults);
  return mlir::success();
}

static mlir::LogicalResult pushDownProjection(SelectOp op,
                                              mlir::PatternRewriter &rewriter) {
  // Find an input that is a projection.
  for (auto &input : op->getOpOperands()) {
    auto defOp = input.get().getDefiningOp();
    if (defOp && defOp->hasTrait<IsProjection>()) {
      assert(defOp->getNumResults() == 1);
      return pushDownProjection(op, input.getOperandNumber(), rewriter);
    }
  }

  return mlir::failure();
}

// Removes input columns that have no use in either the predicates or the
// result.
static mlir::LogicalResult removeUnusedInputs(SelectOp op,
                                              mlir::PatternRewriter &rewriter) {
  // Find inputs that can be removed
  llvm::SmallVector<unsigned int> toRemove;
  auto &body = op.getPredicates().front();
  for (auto idx : llvm::seq(op.getInputs().size())) {
    if (!op.getResult(idx).use_empty() || !body.getArgument(idx).use_empty()) {
      // Result is used.
      continue;
    }

    toRemove.push_back(idx);
  }

  if (toRemove.empty()) {
    // Nothing to remove
    return mlir::failure();
  }

  // Remove in reverse so the indices remain stable.
  llvm::SmallVector<mlir::Value> newInputs(op.getInputs());
  llvm::SmallVector<mlir::Value> replacedResults(op.getResults());
  for (auto it = toRemove.rbegin(); it != toRemove.rend(); ++it) {
    newInputs.erase(newInputs.begin() + *it);
    replacedResults.erase(replacedResults.begin() + *it);

    body.eraseArgument(*it);
  }

  auto newOp = rewriter.create<SelectOp>(
      op.getLoc(), mlir::ValueRange{newInputs}.getTypes(), newInputs);
  rewriter.inlineRegionBefore(op.getPredicates(), newOp.getPredicates(),
                              newOp.getPredicates().end());
  rewriter.replaceAllUsesWith(replacedResults, newOp->getResults());
  rewriter.eraseOp(op);
  return mlir::success();
}

static mlir::LogicalResult
remapDuplicateInputs(SelectOp op, mlir::PatternRewriter &rewriter) {
  // Look for duplicates
  llvm::SmallDenseMap<mlir::Value, unsigned int> remapColumns;

  bool didRewrite = false;
  for (auto &input : op->getOpOperands()) {
    auto [it, newlyAdded] =
        remapColumns.insert({input.get(), input.getOperandNumber()});
    if (newlyAdded) {
      // Not duplicate
      continue;
    }

    // Duplicate
    auto fromIdx = input.getOperandNumber();
    auto toIdx = it->second;
    assert(toIdx < fromIdx);

    // Replace all uses of the argument in predicates
    auto &body = op.getPredicates().front();
    if (!body.getArgument(fromIdx).use_empty()) {
      rewriter.replaceAllUsesWith(body.getArgument(fromIdx),
                                  body.getArgument(toIdx));
      didRewrite = true;
    }

    // Replace uses of the result
    if (!op->getResult(fromIdx).use_empty()) {
      rewriter.replaceAllUsesWith(op->getResult(fromIdx), op->getResult(toIdx));
      didRewrite = true;
    }
  }

  return mlir::success(didRewrite);
}

static mlir::LogicalResult mergeSelect(SelectOp op,
                                       mlir::PatternRewriter &rewriter) {
  // All inputs derive from one SelectOp.
  auto selectOp = op.getInputs()[0].getDefiningOp<SelectOp>();
  if (!selectOp || !llvm::all_of(op.getInputs(), isDefinedBy(selectOp))) {
    return mlir::failure();
  }

  // Create an op combining the two original ops.
  auto newOp = llvm::cast<SelectOp>(rewriter.clone(*selectOp));

  PredicateArgMapping mapping(op, selectOp);
  llvm::SmallVector<PredicateOp> preds(
      op.getPredicates().front().getOps<PredicateOp>());
  for (auto pred : preds) {
    assert(mapping.canPushDown(pred));
    mapping.movePredicate(pred, newOp, rewriter);
  }

  // Replace the original op.
  llvm::SmallVector<mlir::Value> replacedResults;
  for (auto i : llvm::seq(op->getNumResults())) {
    replacedResults.push_back(newOp->getResult(mapping.mapping.at(i)));
  }

  rewriter.replaceOp(op, replacedResults);

  return mlir::success();
}

void PushDownPredicates::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(pushDownAggregate);
  patterns.add(pushDownJoin);
  patterns.add(pushDownUnion);
  patterns.add(pushDownProjection);

  patterns.add(remapDuplicateInputs);
  patterns.add(removeUnusedInputs);
  patterns.add(mergeSelect);

  if (mlir ::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                       std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar