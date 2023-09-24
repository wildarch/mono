add_llvm_executable(query-opt query-opt.cpp)

llvm_update_compile_flags(query-opt)
target_link_libraries(query-opt PRIVATE engine)

mlir_check_all_link_libraries(query-opt)
