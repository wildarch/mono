add_mlir_dialect_library(MLIROperator
        OperatorDialect.cpp
        OperatorOps.cpp
        OperatorTypes.cpp

        DEPENDS
        MLIROperatorOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
)
target_include_directories(
        MLIROperator
PUBLIC
        ${PROJECT_SOURCE_DIR}/include

        # TableGen generated code
        ${PROJECT_BINARY_DIR}/include

        ${MLIR_INCLUDE_DIRS}
)