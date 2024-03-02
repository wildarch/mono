include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})

add_mlir_dialect(PhysicalPlanOps physicalplan)
add_mlir_doc(PhysicalPlanDialect PhysicalPlanDialect PhysicalPlan/ -gen-dialect-doc)
add_mlir_doc(PhysicalPlanOps PhysicalPlanOps PhysicalPlan/ -gen-op-doc)
