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

  bool canPushDown(mlir::Block &block);

  void movePredicate(mlir::Block &oldBlock, mlir::Block &newBlock,
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

bool PredicateArgMapping::canPushDown(mlir::Block &block) {
  for (auto arg : block.getArguments()) {
    if (!arg.use_empty() && !mapping.contains(arg.getArgNumber())) {
      // Argument is used and not included in the mapping.
      return false;
    }
  }

  // All used arguments can be mapped to inputs.
  return true;
}

// Inlines oldBlock into newBlock, using the mapping to remap block arguments
// accordingly.
void PredicateArgMapping::movePredicate(mlir::Block &oldBlock,
                                        mlir::Block &newBlock,
                                        mlir::PatternRewriter &rewriter) {
  // For arguments that are not remapped, and are assumed to have no uses, we
  // still need to have some dummy value that we can use as a replacement. The
  // easiest way to ensure the types line up is to start from the old block
  // argument. These will become invalid after the inlining, but that is okay so
  // long as those argument have no uses inside the block.
  llvm::SmallVector<mlir::Value> argReplacements(oldBlock.getArguments());
  for (auto [oldIdx, newIdx] : mapping) {
    assert(oldBlock.getArgument(oldIdx).getType() ==
           newBlock.getArgument(newIdx).getType());
    argReplacements[oldIdx] = newBlock.getArgument(newIdx);
  }

  rewriter.inlineBlockBefore(&oldBlock, &newBlock, newBlock.end(),
                             argReplacements);
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
                                             mlir::Block &block,
                                             mlir::PatternRewriter &rewriter) {
  PredicateArgMapping predMap(selectOp, aggOp);
  if (!predMap.canPushDown(block)) {
    return mlir::failure();
  }

  // Insert the new select over the aggregation inputs.
  rewriter.setInsertionPoint(aggOp);
  auto newSelect = rewriter.create<SelectOp>(
      selectOp.getLoc(), aggOp.getOperandTypes(), aggOp->getOperands());

  predMap.movePredicate(block, newSelect.addPredicate(), rewriter);

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
  for (auto &block : op.getPredicates()) {
    // Can we move it to before the AggregateOp?
    if (mlir::succeeded(pushDownAggregate(op, aggOp, block, rewriter))) {
      return mlir::success();
    }
  }

  return mlir::failure();
}

static mlir::LogicalResult pushDownJoinLHS(SelectOp selectOp, JoinOp joinOp,
                                           mlir::Block &block,
                                           mlir::PatternRewriter &rewriter) {
  PredicateArgMapping predMap(selectOp, joinOp, JoinSide::LHS);
  if (!predMap.canPushDown(block)) {
    return mlir::failure();
  }

  // New select over the LHS inputs
  auto newSelect = rewriter.create<SelectOp>(
      selectOp.getLoc(), joinOp.getLhs().getTypes(), joinOp.getLhs());

  predMap.movePredicate(block, newSelect.addPredicate(), rewriter);

  // Create new join.
  auto newJoinOp = rewriter.create<JoinOp>(
      joinOp.getLoc(), newSelect->getResults(), joinOp.getRhs());

  // Update existing select to use the new join as input.
  replaceChild(selectOp, joinOp, newJoinOp, rewriter);

  return mlir::success();
}

static mlir::LogicalResult pushDownJoinRHS(SelectOp selectOp, JoinOp joinOp,
                                           mlir::Block &block,
                                           mlir::PatternRewriter &rewriter) {
  PredicateArgMapping predMap(selectOp, joinOp, JoinSide::RHS);
  if (!predMap.canPushDown(block)) {
    return mlir::failure();
  }

  // New select over the RHS inputs
  auto newSelect = rewriter.create<SelectOp>(
      selectOp.getLoc(), joinOp.getRhs().getTypes(), joinOp.getRhs());

  predMap.movePredicate(block, newSelect.addPredicate(), rewriter);

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
  for (auto &block : op.getPredicates()) {
    // Can we move it to one of the join children?
    if (mlir::succeeded(pushDownJoinLHS(op, joinOp, block, rewriter))) {
      return mlir::success();
    } else if (mlir::succeeded(pushDownJoinRHS(op, joinOp, block, rewriter))) {
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

  auto newSelect = rewriter.create<SelectOp>(op.getLoc(), newInputs);
  rewriter.inlineRegionBefore(op.getPredicates(), newSelect.getPredicates(),
                              newSelect.getPredicates().end());
  // Add arguments for the operands we added
  for (auto &block : newSelect.getPredicates()) {
    for (auto input : projInputs) {
      block.addArgument(input.getType(), input.getLoc());
    }

    // Recreate the op inside of predicates if needed.
    if (!block.getArgument(operandIdx).use_empty()) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      mlir::IRMapping projMapping;
      projMapping.map(projInputs,
                      block.getArguments().take_back(projInputs.size()));
      auto *newProj = rewriter.clone(*projOp, projMapping);
      rewriter.replaceAllUsesWith(block.getArgument(operandIdx),
                                  newProj->getResult(0));
    }
  }

  // Recreate the op after the select
  auto newProjInputs = newSelect->getResults().take_back(projInputs.size());
  mlir::IRMapping projMapping;
  projMapping.map(projInputs, newProjInputs);
  auto newProj = rewriter.clone(*projOp, projMapping);

  // Replace uses
  llvm::SmallVector<mlir::Value> newResults(
      newSelect->getResults().take_front(op->getNumResults()));
  newResults[operandIdx] = newProj->getResult(0);
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
  for (auto idx : llvm::seq(op.getInputs().size())) {
    if (!op.getResult(idx).use_empty()) {
      // Result is used.
      continue;
    }

    bool needForPred = false;
    for (auto &pred : op.getPredicates()) {
      if (!pred.getArgument(idx).use_empty()) {
        // Needed for predicate evaluation
        needForPred = true;
        break;
      }
    }

    if (needForPred) {
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

    for (auto &pred : op.getPredicates()) {
      pred.eraseArgument(*it);
    }
  }

  auto newOp = rewriter.create<SelectOp>(
      op.getLoc(), mlir::ValueRange{newInputs}.getTypes(), newInputs);
  rewriter.inlineRegionBefore(op.getPredicates(), newOp.getPredicates(),
                              newOp.getPredicates().end());
  rewriter.replaceAllUsesWith(replacedResults, newOp->getResults());
  rewriter.eraseOp(op);
  return mlir::success();
}

// If two of the inputs are equivalent, remaps all uses to the first one.
// Removal of the redundant input is handled by removeUnusedInputs.
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
    for (auto &pred : op.getPredicates()) {
      if (pred.getArgument(fromIdx).use_empty()) {
        // Not used.
        continue;
      }

      // Replace with equivalent block argument.
      rewriter.replaceAllUsesWith(pred.getArgument(fromIdx),
                                  pred.getArgument(toIdx));
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

void PushDownPredicates::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  // NOTE: must be higher priority than pushDownProjection to avoid pulling up
  // the same projection multiple times.
  patterns.add(removeUnusedInputs);
  patterns.add(remapDuplicateInputs);

  patterns.add(pushDownAggregate);
  patterns.add(pushDownJoin);
  patterns.add(pushDownUnion);
  patterns.add(pushDownProjection);

  if (mlir ::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                       std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar