# See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html
cmake_minimum_required(VERSION 4.0)

project(DBLang
    VERSION 1.0
    DESCRIPTION "Systems programming language for DB implementation"
    LANGUAGES CXX)

include_directories(.)

add_library(
    dblang
    parse/Lexer.cpp
    parse/Location.cpp
    util/ReportError.cpp
    util/FileSystem.cpp
)
target_compile_features(dblang PRIVATE cxx_std_17)

add_executable(
    main
    main.cpp
)
target_link_libraries(main dblang)
target_compile_features(main PRIVATE cxx_std_17)

add_executable(
    LexerTest
    parse/LexerTest.cpp
)
target_link_libraries(LexerTest dblang)
target_compile_features(LexerTest PRIVATE cxx_std_17)