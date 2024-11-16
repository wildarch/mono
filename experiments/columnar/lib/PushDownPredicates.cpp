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

static std::optional<unsigned int> inputIdxForResult(JoinOp op, mlir::Value v) {
  for (auto result : op.getResults()) {
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
  inputMapping.map(aggOp->getResults(), newAgg->getResults());
  rewriter.modifyOpInPlace(selectOp, [&]() {
    llvm::SmallVector<mlir::Value> newOperands;
    for (auto v : selectOp.getOperands()) {
      newOperands.push_back(inputMapping.lookup(v));
    }

    selectOp->setOperands(newOperands);
  });

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

static mlir::LogicalResult pushDownJoin(SelectOp selectOp, JoinOp joinOp,
                                        mlir::Block &block,
                                        mlir::PatternRewriter &rewriter) {
  // We can do the push-down if all predicate inputs are on one side of the
  // join.
  bool needLhs = false;
  bool needRhs = false;

  auto nLhs = joinOp.getLhs().size();
  for (auto arg : block.getArguments()) {
    if (arg.use_empty()) {
      // Unused arg
      continue;
    }

    auto joinInputIdx =
        inputIdxForResult(joinOp, selectOp->getOperand(arg.getArgNumber()));
    if (*joinInputIdx < nLhs) {
      needLhs = true;
    } else {
      needRhs = true;
    }
  }

  if (needLhs && needRhs) {
    // Need both sides, cannot push down
    return mlir::failure();
  }

  if (!needRhs) {
    // Push down to LHS

    // New select over the LHS inputs
    auto newSelect = rewriter.create<SelectOp>(
        selectOp.getLoc(), joinOp.getLhs().getTypes(), joinOp.getLhs());

    // Map the old block args to the new ones, where possible.
    auto &newBlock = newSelect.addPredicate();
    llvm::SmallVector<mlir::Value> argReplacements(block.getArguments());
    for (auto oldArg : block.getArguments()) {
      // The input column for the old argument
      auto oldInput = selectOp.getInputs()[oldArg.getArgNumber()];
      // Which input of the join side, and therefore of the new select op, is
      // mapped to the old argument.
      auto newInputIdx = inputIdxForResult(joinOp, oldInput);
      if (!newInputIdx) {
        // No replacement, should not be used.
        assert(oldArg.use_empty());
        continue;
      }

      argReplacements[oldArg.getArgNumber()] =
          newBlock.getArgument(*newInputIdx);
    }

    // Move predicate to the select BEFORE the join
    rewriter.inlineBlockBefore(&block, &newBlock, newBlock.end(),
                               argReplacements);

    auto newJoinOp = rewriter.create<JoinOp>(
        joinOp.getLoc(), newSelect->getResults(), joinOp.getRhs());

    // Update select to use the new join op
    rewriter.modifyOpInPlace(
        selectOp, [&]() { selectOp->setOperands(newJoinOp->getResults()); });
    return mlir::success();
  } else {
    assert(!needLhs);

    return selectOp->emitOpError("Push down to RHS not implemented");
  }
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
    if (mlir::succeeded(pushDownJoin(op, joinOp, block, rewriter))) {
      return mlir::success();
    }
  }

  return mlir::failure();
}

void PushDownPredicates::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add(pushDownAggregate);
  patterns.add(pushDownJoin);

  if (mlir ::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                       std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace columnar