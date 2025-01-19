#include "columnar/Columnar.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace columnar {

#define GEN_PASS_DEF_MERGECHUNKS
#include "columnar/Passes.h.inc"

namespace {

class MergeChunks : public impl::MergeChunksBase<MergeChunks> {
public:
  using impl::MergeChunksBase<MergeChunks>::MergeChunksBase;

  void runOnOperation() final;
};

} // namespace

void MergeChunks::runOnOperation() {
  auto pipelineOp = getOperation();
  auto &body = pipelineOp.getBody().front();

  mlir::IRRewriter rewriter(&getContext());
  rewriter.setInsertionPointToEnd(&body);

  llvm::SmallVector<ChunkOp> toMerge(pipelineOp.getOps<ChunkOp>());

  auto mergedOp = rewriter.create<ChunkOp>(
      pipelineOp.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
  auto &mergedBody = mergedOp.getBody().front();

  mlir::IRMapping mapping;
  for (auto chunkOp : toMerge) {
    auto &body = chunkOp.getBody().front();

    // Map chunk inputs to already-inlined values.
    llvm::SmallVector<mlir::Value> inputs;
    for (auto input : chunkOp.getInputs()) {
      inputs.emplace_back(mapping.lookup(input));
    }

    // Inline the chunk
    rewriter.inlineBlockBefore(&body, &mergedBody, mergedBody.end(), inputs);

    // Map chunk yield inputs to the chunk results.
    auto yieldOp = llvm::cast<ChunkYieldOp>(mergedBody.getTerminator());
    for (auto [input, result] :
         llvm::zip_equal(yieldOp.getInputs(), chunkOp->getResults())) {
      mapping.map(result, input);
    }

    rewriter.eraseOp(yieldOp);
  }

  // Terminate the merged chunk.
  rewriter.setInsertionPointToEnd(&mergedBody);
  rewriter.create<ChunkYieldOp>(pipelineOp.getLoc(), mlir::ValueRange{});

  for (auto it = toMerge.rbegin(); it != toMerge.rend(); it++) {
    auto chunkOp = *it;
    rewriter.eraseOp(chunkOp);
  }
}

} // namespace columnar