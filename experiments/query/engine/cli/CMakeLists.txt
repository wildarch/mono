add_executable(cli main.cpp)
target_link_libraries(cli engine)
set_target_properties(
    cli
PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
