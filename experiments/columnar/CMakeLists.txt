# See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html
cmake_minimum_required(VERSION 4.0)

project(Columnar
    VERSION 1.0
    DESCRIPTION "Columnar, compiled query execution engine"
    LANGUAGES CXX)

# LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)

llvm_map_components_to_libnames(llvm_libs support)

# MLIR
find_package(MLIR ${LLVM_PACKAGE_VERSION} REQUIRED CONFIG
    HINTS "${LLVM_INSTALL_PREFIX}/lib/cmake"
)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# Directories with MLIR dialects
add_subdirectory(include/columnar)

add_subdirectory(lib)

add_executable(
    execute
    execute.cpp
)
target_compile_features(execute PRIVATE cxx_std_17)
target_link_libraries(execute
    PRIVATE
    columnar
    MLIRExecutionEngine
)

add_executable(
    columnar-opt
    columnar-opt.cpp
)
target_compile_features(columnar-opt PRIVATE cxx_std_17)
target_link_libraries(columnar-opt
    PRIVATE
    columnar
    MLIROptLib
)
