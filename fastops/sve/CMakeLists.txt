set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve -Wno-unknown-pragmas -Wno-unused-local-typedef")

add_library(sve ops_sve.cpp)
