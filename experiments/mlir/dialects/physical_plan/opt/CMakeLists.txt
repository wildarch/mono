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
target_compile_options(opt-physicalplan PRIVATE -fsanitize=address)
target_link_options(opt-physicalplan PRIVATE -fsanitize=address)

mlir_check_all_link_libraries(opt-physicalplan)
