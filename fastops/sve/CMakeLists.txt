add_library(sve ops_sve.cpp)
target_compile_options(sve PRIVATE -march=armv8-a+sve -Wno-unknown-pragmas -Wno-unused-local-typedef)
