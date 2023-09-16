# based on https://cliutils.gitlab.io/modern-cmake/
cmake_minimum_required(VERSION 3.22)

project(engine VERSION 0.1
               DESCRIPTION "Query Engine"
               LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

add_executable(engine main.cpp)
target_link_libraries(
    engine 
    Arrow::arrow_static
    Parquet::parquet_static
)