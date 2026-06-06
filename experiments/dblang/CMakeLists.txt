# See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html
cmake_minimum_required(VERSION 4.0)

project(DBLang
    VERSION 1.0
    DESCRIPTION "Systems programming language for DB implementation"
    LANGUAGES CXX)

include_directories(.)

add_executable(
    main
    main.cpp
    parse/Lexer.cpp
    util/ReportError.cpp
)
target_compile_features(main PRIVATE cxx_std_17)