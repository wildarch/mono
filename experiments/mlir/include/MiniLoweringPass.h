#include <memory>

namespace mlir {
class Pass;
}

namespace experiments_mlir::mini {
std::unique_ptr<mlir::Pass> createMiniLoweringPass();
}