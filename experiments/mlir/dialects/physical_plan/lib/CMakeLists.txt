add_mlir_dialect_library(MLIRPhysicalPlan
        PhysicalPlanDialect.cpp
        PhysicalPlanOps.cpp
        PhysicalPlanTypes.cpp
        VectorizeCompute.cpp
        VectorizePipelines.cpp
        PlanToSCF.cpp
        PlanToLLVM.cpp

        DEPENDS
        MLIRPhysicalPlanOpsIncGen
        MLIRPhysicalPlanPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRVectorDialect
        MLIRSCFDialect
        MLIRLLVMDialect
        MLIRArithToLLVM
        MLIRSCFToControlFlow
        MLIRControlFlowToLLVM
        MLIRFuncDialect
        MLIRFuncToLLVM
)
target_include_directories(
        MLIRPhysicalPlan
PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        
        # TableGen generated code
        ${PROJECT_BINARY_DIR}/include

        ${MLIR_INCLUDE_DIRS}
)