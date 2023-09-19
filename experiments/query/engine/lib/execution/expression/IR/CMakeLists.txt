add_mlir_dialect_library(MLIRExpression
        ExpressionDialect.cpp
        ExpressionOps.cpp

        DEPENDS
        MLIRExpressionOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
)
target_include_directories(
        MLIRExpression
PRIVATE
        # TableGen generated code
        ${PROJECT_BINARY_DIR}/include
PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${MLIR_INCLUDE_DIRS}
)