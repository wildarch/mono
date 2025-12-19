# https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#introduction-to-the-basics
cmake_minimum_required(VERSION 3.15...4.0)

project(
    HighPerfArithExpr
    VERSION 0.1
    DESCRIPTION "High-performance Arithmetic Expression"
    LANGUAGES CXX)


add_executable(
    gen
    gen.cpp
)
