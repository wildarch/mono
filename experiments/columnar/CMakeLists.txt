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

# Protobuf
find_package(Protobuf REQUIRED)

# pg_query (See https://dominikberner.ch/cmake-find-library/)
find_library(PG_QUERY_LIBRARY pg_query REQUIRED)
find_path(PG_QUERY_INCLUDE_DIR NAMES pg_query.h)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    libpg_query DEFAULT_MSG
    PG_QUERY_LIBRARY
    PG_QUERY_INCLUDE_DIR)
mark_as_advanced(PG_QUERY_LIBRARY PG_QUERY_INCLUDE_DIR)
add_library(libpg_query::libpg_query SHARED IMPORTED)
set_target_properties(
    libpg_query::libpg_query
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PG_QUERY_INCLUDE_DIR}"
    IMPORTED_LOCATION ${PG_QUERY_LIBRARY})

# Parquet
find_package(Parquet REQUIRED)

# Directories with MLIR dialects
add_subdirectory(include/columnar)

add_subdirectory(lib)
add_subdirectory(test)

add_executable(
    translate
    translate.cpp
)
target_compile_features(translate PRIVATE cxx_std_17)
target_link_libraries(translate
    PRIVATE
    columnar
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
    parquet
    parquet.cpp
)
target_compile_features(parquet PRIVATE cxx_std_17)
target_link_libraries(parquet
    PRIVATE
    columnar
    Parquet::parquet_shared
)
