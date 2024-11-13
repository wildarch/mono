#include "columnar/Columnar.h"

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

} // namespace

static auto isDefinedBy(mlir::Operation *op) {
  return [op](mlir::Value v) { return v.getDefiningOp() == op; };
}

static bool isGroupByResult(AggregateOp op, mlir::Value v) {
  auto groupByResults = op->getResults().take_front(op.getGroupBy().size());
  return llvm::is_contained(groupByResults, v);
}

static mlir::LogicalResult pushDownAggregate(SelectOp selectOp,
                                             AggregateOp aggOp,
                                             mlir::Block &block,
                                             mlir::PatternRewriter &rewriter) {
  // 1. Skip args that are not used
  // 2. Check that the column is an aggregate GROUP_BY result
  // 3. Find the corresponding column on the AggregateOp input.
  // 4. Make a new SelectOp with this predicate, and set it over the inputs of
  // the aggregate.
  for (auto [input, arg] :
       llvm::zip_equal(selectOp.getInputs(), block.getArguments())) {
    if (arg.use_empty()) {
      // Skip unused columns.
      continue;
    }

    if (!isGroupByResult(aggOp, input)) {
      // We can only push down predicates if they exclusively depend on the
      // group-by keys.
      return mlir::failure();
    }
  }

  // Good to go!

  // Insert the new select over the aggregation inputs.
  rewriter.setInsertionPoint(aggOp);
  auto newOp = rewriter.create<SelectOp>(
      selectOp.getLoc(), aggOp.getOperandTypes(), aggOp->getOperands());

  // Update aggregation to use the new select as input.
  rewriter.modifyOpInPlace(aggOp,
                           [&]() { aggOp->setOperands(newOp->getResults()); });

  // Map the old block args to the new ones, where possible.
  llvm::SmallVector<mlir::Value> argReplacements;

  auto &newBlock = newOp.getPredicates().emplaceBlock();
  for (auto [input, oldArg] :
       llvm::zip_equal(newOp.getInputs(), block.getArguments())) {
    // Create the new block arguments
    auto newArg = newBlock.addArgument(input.getType(), input.getLoc());

    // Map old arguments to new ones.
    if (newArg.getType() == oldArg.getType()) {
      argReplacements.emplace_back(newArg);
    } else {
      assert(oldArg.use_empty());
      // Need to put something with the correct type.
      // Will not be used.
      argReplacements.push_back(oldArg);
    }
  }

  // Move predicate to the select BEFORE the aggregation
  rewriter.inlineBlockBefore(&block, &newBlock, newBlock.end(),
                             argReplacements);
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

void PushDownPredicates::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(pushDownAggregate);

  if (mlir ::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                       std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar