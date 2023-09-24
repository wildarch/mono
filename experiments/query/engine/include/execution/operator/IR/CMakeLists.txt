include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})

add_mlir_dialect(OperatorOps operator)
add_mlir_doc(OperatorDialect OperatorDialect Operator/ -gen-dialect-doc)
add_mlir_doc(OperatorOps OperatorOps Operator/ -gen-op-doc)
