if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    add_subdirectory(neon)
    add_subdirectory(sve)
    add_subdirectory(plain)

    add_library(fastops fastops.cpp)
    target_link_libraries(fastops neon sve plain)
else()
    add_subdirectory(avx)
    add_subdirectory(avx2)
    add_subdirectory(plain)

    add_library(fastops fastops.cpp core/avx_id.cpp)
    target_link_libraries(fastops avx avx2 plain)
endif()

install(TARGETS fastops DESTINATION lib)
install(FILES fastops.h DESTINATION include)
install(FILES core/FastIntrinsics.h core/SIMDFunctions.h core/avx_id.h DESTINATION include/core)
install(FILES plain/ops_plain.h DESTINATION include/plain)

add_subdirectory(ut)
