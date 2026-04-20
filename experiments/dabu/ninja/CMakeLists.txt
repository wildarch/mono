# See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html
cmake_minimum_required(VERSION 4.0)

project(DabuNinja
    VERSION 1.0
    DESCRIPTION "Ninja compat layer for dabu build system"
    LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  Ninja
  SOURCE_DIR ../thirdparty/ninja-1.13.2
)
FetchContent_MakeAvailable(Ninja)

# Boost.JSON
find_package(Boost REQUIRED)

add_executable(
    dabu-ninja
    dabu-ninja.cpp
)
target_link_libraries(dabu-ninja PRIVATE
    libninja
    libninja-re2c)
