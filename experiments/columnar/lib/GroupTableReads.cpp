#include <mlir/IR/PatternMatch.h>

#include "columnar/Columnar.h"

namespace columnar {

#define GEN_PASS_DEF_GROUPTABLEREADS
#include "columnar/Passes.h.inc"

namespace {

class GroupTableReads : public impl::GroupTableReadsBase<GroupTableReads> {
public:
  using impl::GroupTableReadsBase<GroupTableReads>::GroupTableReadsBase;

  void runOnOperation() final;
};

} // namespace

static void runOnPipeline(PipelineOp pipe, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<ReadColumnOp> readColumnOps;
  pipe->walk([&](ReadColumnOp op) { readColumnOps.push_back(op); });

  llvm::SmallVector<SelTableOp> selTableOps;
  pipe->walk([&](SelTableOp op) { selTableOps.push_back(op); });
  if (selTableOps.size() != 1) {
    return;
  }

  SelTableOp selTableOp = selTableOps.front();

  llvm::SmallVector<mlir::Type> columnTypes{
      selTableOp.getType(),
  };
  llvm::SmallVector<TableColumnAttr> columns;
  for (auto op : readColumnOps) {
    columnTypes.push_back(op.getType());
    columns.push_back(op.getColumn());
  }

  rewriter.setInsertionPointToStart(&pipe.getBody().front());
  auto readTableOp = rewriter.create<ReadTableOp>(
      selTableOp.getLoc(), columnTypes, selTableOp.getTable(), columns);
  // Replace single column read ops and selection vector.
  rewriter.replaceOp(selTableOp, readTableOp.getSel());
  for (auto [op, replacement] :
       llvm::zip_equal(readColumnOps, readTableOp.getCol())) {
    rewriter.replaceOp(op, replacement);
  }
}

void GroupTableReads::runOnOperation() {
  mlir::IRRewriter rewriter(&getContext());
  getOperation()->walk([&](PipelineOp op) { runOnPipeline(op, rewriter); });
}

} // namespace columnar
