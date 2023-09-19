include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})

add_mlir_dialect(ExpressionOps expression)
add_mlir_doc(ExpressionDialect ExpressionDialect Expression/ -gen-dialect-doc)
add_mlir_doc(ExpressionOps ExpressionOps Expression/ -gen-op-doc)
