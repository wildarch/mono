include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_llvm_executable(opt-physicalplan opt.cpp)

llvm_update_compile_flags(opt-physicalplan)
target_link_libraries(opt-physicalplan 
    PRIVATE 
    MLIRPhysicalPlan 
    MLIROptLib
    MLIRArithDialect
)

mlir_check_all_link_libraries(opt-physicalplan)
