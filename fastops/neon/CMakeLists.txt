add_library(neon ops_neon.cpp)
target_compile_options(neon PRIVATE -Wno-unknown-pragmas -Wno-unused-local-typedef)
