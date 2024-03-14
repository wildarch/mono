#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanPasses.h"

int main(int argc, char **argv) {
  // TODO: Register custom passes here.
  // physicalplan::PhysicalPlan::registerPasses();
  // Or all the built-ins
  // mlir::registerAllPasses();
  physicalplan::registerPasses();
  mlir::registerCanonicalizer();

  mlir::DialectRegistry registry;
  registry.insert<physicalplan::PhysicalPlanDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::func::FuncDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Optimizer driver for PhysicalPlan dialect\n", registry));
}
