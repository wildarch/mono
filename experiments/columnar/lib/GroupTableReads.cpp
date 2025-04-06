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

void GroupTableReads::runOnOperation() {
  llvm::SmallVector<ReadColumnOp> readColumnOps;
  getOperation()->walk([&](ReadColumnOp op) { readColumnOps.push_back(op); });

  llvm::SmallVector<SelTableOp> selTableOps;
  getOperation()->walk([&](SelTableOp op) { selTableOps.push_back(op); });
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

  mlir::IRRewriter rewriter(&getContext());
  rewriter.setInsertionPointToStart(&getOperation().getBody().front());
  auto readTableOp = rewriter.create<ReadTableOp>(
      selTableOp.getLoc(), columnTypes, selTableOp.getTable(), columns);
  // Replace single column read ops and selection vector.
  rewriter.replaceOp(selTableOp, readTableOp.getSel());
  for (auto [op, replacement] :
       llvm::zip_equal(readColumnOps, readTableOp.getCol())) {
    rewriter.replaceOp(op, replacement);
  }
}

} // namespace columnar