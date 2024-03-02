# based on https://cliutils.gitlab.io/modern-cmake/
cmake_minimum_required(VERSION 3.22)

project(engine VERSION 0.1
               DESCRIPTION "TODO: write a description"
               LANGUAGES CXX C)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_package(LLVM 17 REQUIRED CONFIG)
include(TableGen)
include(AddLLVM)

# Use LLVM config to find MLIR
if (NOT DEFINED MLIR_DIR)
    set(MLIR_DIR ${LLVM_INSTALL_PREFIX}/lib/cmake/mlir)
endif()
find_package(MLIR 17 REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddMLIR)
include(HandleLLVMOptions)

# MLIR dialect
add_subdirectory(include)

# Library
add_subdirectory(lib)

# Binaries
add_subdirectory(opt)