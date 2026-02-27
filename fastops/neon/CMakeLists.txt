set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-local-typedef")

add_library(neon ops_neon.cpp)
