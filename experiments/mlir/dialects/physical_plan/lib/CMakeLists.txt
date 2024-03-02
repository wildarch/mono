add_mlir_dialect_library(MLIRPhysicalPlan
        PhysicalPlanDialect.cpp
        PhysicalPlanOps.cpp

        DEPENDS
        MLIRPhysicalPlanOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
)
target_include_directories(
        MLIRPhysicalPlan
PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        
        # TableGen generated code
        ${PROJECT_BINARY_DIR}/include

        ${MLIR_INCLUDE_DIRS}
)