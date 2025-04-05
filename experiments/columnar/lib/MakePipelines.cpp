#include "columnar/Columnar.h"

namespace columnar {

#define GEN_PASS_DEF_MAKEPIPELINES
#include "columnar/Passes.h.inc"

namespace {

class MakePipelines : public impl::MakePipelinesBase<MakePipelines> {
public:
  using impl::MakePipelinesBase<MakePipelines>::MakePipelinesBase;

  void runOnOperation() final;
};

} // namespace

static mlir::Value createInPipeline(mlir::Value v, mlir::IRMapping &mapping,
                                    mlir::IRRewriter &rewriter) {
  if (mapping.contains(v)) {
    return mapping.lookup(v);
  } else if (auto op = v.getDefiningOp()) {
    for (auto oper : op->getOperands()) {
      createInPipeline(oper, mapping, rewriter);
    }

    auto *clone = rewriter.clone(*op, mapping);
    mapping.map(op, clone);
    return mapping.lookup(v);
  } else {
    mlir::emitError(v.getLoc()) << "Cannot create in pipeline: " << v;
    return nullptr;
  }
}

static void makePipeline(mlir::Operation *sink, mlir::IRRewriter &rewriter) {
  auto pipeOp = rewriter.create<PipelineOp>(sink->getLoc());
  auto &pipeBody = pipeOp.getBody().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&pipeBody);

  llvm::SmallVector<mlir::Value> newOperands;
  mlir::IRMapping mapping;
  for (auto oper : sink->getOperands()) {
    newOperands.emplace_back(createInPipeline(oper, mapping, rewriter));
  }

  rewriter.moveOpAfter(sink, &pipeBody, pipeBody.begin());
  rewriter.modifyOpInPlace(sink, [&]() { sink->setOperands(newOperands); });
}

void MakePipelines::runOnOperation() {
  // Pure sinks.
  llvm::SmallVector<mlir::Operation *> sinks;
  // Ops that are both source and sink need to be split up.
  llvm::SmallVector<mlir::Operation *> breakers;

  getOperation()->walk([&](mlir::Operation *op) {
    if (op->hasTrait<Sink>()) {
      if (op->hasTrait<Source>()) {
        breakers.emplace_back(op);
      } else {
        sinks.emplace_back(op);
      }
    }
  });

  if (!breakers.empty()) {
    for (auto *op : breakers) {
      op->emitOpError("Cannot split pipeline breaker");
    }

    return signalPassFailure();
  }

  mlir::IRRewriter rewriter(&getContext());
  rewriter.setInsertionPointToEnd(getOperation().getBody());
  for (auto *op : sinks) {
    makePipeline(op, rewriter);
  }

  // Clean up QueryOps that now don't have terminators anymore.
  llvm::SmallVector<QueryOp> queryOps;
  getOperation()->walk([&](QueryOp op) { queryOps.emplace_back(op); });
  for (auto op : queryOps) {
    rewriter.eraseOp(op);
  }
}

} // namespace columnar