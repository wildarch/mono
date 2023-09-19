# MLIR expression dialect
add_subdirectory(execution/expression/IR)

set(
    ENGINE_SOURCE_FILES
    execution/Batch.cpp
    execution/ParquetScanner.cpp
    execution/expression/ExpressionEvaluator.cpp
    execution/operator/ParquetScanOperator.cpp
    execution/operator/FilterOperator.cpp
)
add_library(engine ${ENGINE_SOURCE_FILES})
target_include_directories(
    engine
PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    engine 
    Arrow::arrow_static
    Parquet::parquet_static
    MLIRExpression
)
set_target_properties(
    engine
PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)