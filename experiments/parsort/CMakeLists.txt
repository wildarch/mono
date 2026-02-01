# See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html
cmake_minimum_required(VERSION 4.0)

project(ParSort
    VERSION 1.0
    DESCRIPTION "Parallel External Merge Sort"
    LANGUAGES CXX)

add_executable(
    parsort
    parsort.cpp
)
target_compile_features(parsort PRIVATE cxx_std_17)
