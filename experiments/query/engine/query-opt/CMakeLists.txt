get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    MLIRExpression
    MLIROperator
)
add_llvm_executable(query-opt query-opt.cpp)

llvm_update_compile_flags(query-opt)
target_link_libraries(query-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(query-opt)
