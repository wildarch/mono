include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS PhysicalPlanPasses.td)
add_mlir_dialect(PhysicalPlanOps physicalplan)
add_mlir_doc(PhysicalPlanDialect PhysicalPlanDialect PhysicalPlan/ -gen-dialect-doc)
add_mlir_doc(PhysicalPlanOps PhysicalPlanOps PhysicalPlan/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PhysicalPlanPasses.td)
mlir_tablegen(PhysicalPlanPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRPhysicalPlanPassesIncGen)