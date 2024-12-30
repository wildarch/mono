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

struct WithSel {
  mlir::Value sel;
  llvm::SmallVector<mlir::Value> columns;
};

class SelectionVectorTracker {
private:
  llvm::DenseMap<mlir::Value, mlir::Value> _resolved;

public:
  void createTableReadSelectionVectors(QueryOp query,
                                       mlir::RewriterBase &rewriter);

  bool allInputsResolved(mlir::ValueRange inputs);
  mlir::FailureOr<WithSel> getOrCreateInputs(mlir::ValueRange inputs,
                                             mlir::RewriterBase &rewriter);
};

} // namespace

void SelectionVectorTracker::createTableReadSelectionVectors(
    QueryOp query, mlir::RewriterBase &rewriter) {

  // All tables referenced in the query.
  llvm::SmallDenseSet<mlir::StringAttr> tables;
  query->walk([&](ReadTableOp op) { tables.insert(op.getTableAttr()); });

  // Create selection vectors
  llvm::SmallDenseMap<mlir::StringAttr, mlir::Value> selPerTable;
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&query.getBody().front());
  for (auto table : tables) {
    selPerTable[table] =
        rewriter.create<SelTableOp>(rewriter.getUnknownLoc(), table);
  }

  // Mark as resolved
  query->walk([&](ReadTableOp op) {
    _resolved[op] = selPerTable.at(op.getTableAttr());
  });
}

bool SelectionVectorTracker::allInputsResolved(mlir::ValueRange inputs) {
  return llvm::all_of(inputs, [this](mlir::Value v) {
    return _resolved.contains(v) || v.getDefiningOp<ConstantOp>();
  });
}

mlir::FailureOr<WithSel>
SelectionVectorTracker::getOrCreateInputs(mlir::ValueRange inputs,
                                          mlir::RewriterBase &rewriter) {
  WithSel result{
      .columns = inputs,
  };
  for (auto input : inputs) {
    if (auto inSel = _resolved.lookup(input)) {
      if (!result.sel) {
        result.sel = inSel;
      } else {
        return mlir::emitError(input.getLoc())
               << "input " << input << " has resolved selection vector "
               << inSel
               << ", but the selection vector was previously resolved as "
               << result.sel;
      }
    } else {
      assert(input.getDefiningOp<ConstantOp>());
    }
  }

  if (!result.sel) {
    result.sel = rewriter.create<SelConstOp>(rewriter.getUnknownLoc());
  }

  return result;
}

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

void AddSelectionVectors::runOnOperation() {
  // 1. Add all ops to a worklist.
  llvm::SmallPtrSet<mlir::Operation *, 16> worklist;
  for (auto &op : getOperation().getBody().front()) {
    worklist.insert(&op);
  }

  SelectionVectorTracker tracker;
  mlir::IRRewriter rewriter(getOperation());
  tracker.createTableReadSelectionVectors(getOperation(), rewriter);

  // 2. Iterate on the worklist:
  //   a. find an operation to process: All inputs have to have selection
  //   vectors available.
  //   b. if no operation is found, break out of the loop
  //   c. remove the operation from the worklist
  //   d. rewrite the ops to use selection vectors
  while (!worklist.empty()) {
    auto nextOp = llvm::find_if(worklist, [&](mlir::Operation *op) {
      return tracker.allInputsResolved(op->getOperands());
    });

    if (nextOp == worklist.end()) {
      break;
    }

    auto *op = *nextOp;
    worklist.erase(op);

    // TODO
    op->emitOpError("adding selection vector");
  }

  // 3. If the worklist is empty, the algorithm has completed.
  if (!worklist.empty()) {
    getOperation()->emitOpError("failed to add selection vectors");
    return signalPassFailure();
  }
}

} // namespace columnar