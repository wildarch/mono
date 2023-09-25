# MLIR dialects
add_subdirectory(execution/expression/IR)
add_subdirectory(execution/operator/IR)

set(
    ENGINE_SOURCE_FILES
    execution/Batch.cpp
    execution/expression/bytecode/ExpressionByteCode.cpp
    execution/expression/ExpressionEvaluator.cpp
    execution/operator/impl/ParquetScanOperator.cpp
    execution/operator/impl/FilterOperator.cpp
    execution/ImplementationGenerator.cpp
    execution/ParquetScanner.cpp
)
add_library(engine ${ENGINE_SOURCE_FILES})
target_include_directories(
    engine
PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(
    engine 
    Arrow::arrow_static
    Parquet::parquet_static
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    MLIRExpression
    MLIROperator
)
set_target_properties(
    engine
PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)