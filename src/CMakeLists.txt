set(CMAKE_VERBOSE_MAKEFILE on)
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if (${LINUX})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wfatal-errors -march=native -fopenmp")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wfatal-errors -march=native")
endif ()

set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (${APPLE})
    include_directories(/usr/local/include/) # required by Mac OS to find boost
endif ()

set(SOURCE_FILES util.h competitors/rmi_universal.cpp)
# file(GLOB_RECURSE RMI_SOURCES "competitors/rmi/*.cpp")
# file(GLOB_RECURSE BENCH_SOURCES "benchmarks/*.cc")

# if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
#     # FIXME cannot link with g++-9
#     add_definitions(-DDISABLE_FST=1)
#     list(REMOVE_ITEM BENCH_SOURCES "${CMAKE_SOURCE_DIR}/benchmarks/benchmark_fst.cc")
# endif ()
# 
# if (${APPLE})
#     # Certain headers (e.g., byteswap) not supported under Mac OS.
#     list(REMOVE_ITEM BENCH_SOURCES "${CMAKE_SOURCE_DIR}/benchmarks/benchmark_wormhole.cc")
# endif ()

add_executable(generate_queries generators/generate_queries.cc ${SOURCE_FILES})
target_link_libraries(generate_queries Threads::Threads)

add_executable(benchmark benchmark.cc ${SOURCE_FILES} ${RMI_SOURCES} ${BENCH_SOURCES})
# target_include_directories(benchmark
#        PRIVATE "competitors/stx-btree-0.9/include"
#        PRIVATE "competitors/FST/include"
#        PRIVATE "competitors/PGM-index/include"
#        PRIVATE "competitors/rs/include"
#        PRIVATE "fast64"
# )

target_link_libraries(benchmark
        PRIVATE Threads::Threads
#        dtl
        PRIVATE dl
#        stdc++fs
        )
