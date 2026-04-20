# See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html
cmake_minimum_required(VERSION 4.0)

project(CMakeHello
    VERSION 1.0
    DESCRIPTION "Trivial CMake project to test dabu build tool"
    LANGUAGES CXX)

add_executable(
    hello
    hello.cpp
)