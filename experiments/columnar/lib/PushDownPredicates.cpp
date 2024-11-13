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

static std::optional<unsigned int> inputIdxForResult(AggregateOp op,
                                                     mlir::Value v) {
  for (auto [input, result] : llvm::zip(op.getGroupBy(), op.getResults())) {
    if (v == result) {
      return result.getResultNumber();
    }
  }

  return std::nullopt;
}

static mlir::LogicalResult pushDownAggregate(SelectOp selectOp,
                                             AggregateOp aggOp,
                                             mlir::Block &block,
                                             mlir::PatternRewriter &rewriter) {
  for (auto [input, arg] :
       llvm::zip_equal(selectOp.getInputs(), block.getArguments())) {
    if (arg.use_empty()) {
      // Skip unused columns.
      continue;
    }

    if (!inputIdxForResult(aggOp, input)) {
      // We can only push down predicates if they exclusively depend on
      // propagated columns.
      return mlir::failure();
    }
  }

  // Good to go!

  // Insert the new select over the aggregation inputs.
  rewriter.setInsertionPoint(aggOp);
  auto newOp = rewriter.create<SelectOp>(
      selectOp.getLoc(), aggOp.getOperandTypes(), aggOp->getOperands());

  // Map the old block args to the new ones, where possible.
  auto &newBlock = newOp.addPredicate();
  llvm::SmallVector<mlir::Value> argReplacements(block.getArguments());
  for (auto oldArg : block.getArguments()) {
    // The input column for the old argument
    auto oldInput = selectOp.getInputs()[oldArg.getArgNumber()];
    // Which input of the aggregate, and therefore of the new select op, is
    // mapped to the old argument.
    auto newInputIdx = inputIdxForResult(aggOp, oldInput);
    if (!newInputIdx) {
      // No replacement, should not be used.
      assert(oldArg.use_empty());
      continue;
    }

    argReplacements[oldArg.getArgNumber()] = newBlock.getArgument(*newInputIdx);
  }

  // Move predicate to the select BEFORE the aggregation
  rewriter.inlineBlockBefore(&block, &newBlock, newBlock.end(),
                             argReplacements);

  // Update aggregation to use the new select as input.
  rewriter.modifyOpInPlace(aggOp,
                           [&]() { aggOp->setOperands(newOp->getResults()); });

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