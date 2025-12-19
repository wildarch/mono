# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    HighPerfArithExpr
    VERSION 0.1
    DESCRIPTION "High-performance Arithmetic Expression"
    LANGUAGES CXX)

# Generally an anti-pattern, but useful in this repo.
include_directories(include/)

add_subdirectory(lib)

add_executable(gen gen.cpp)

add_executable(legal_words legal_words.cpp)
target_link_libraries(legal_words PRIVATE hae)
