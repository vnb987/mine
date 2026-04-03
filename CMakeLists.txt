cmake_minimum_required(VERSION 3.18)
project(OnionPIRv2_CUDA LANGUAGES CXX CUDA)

# ============================================================================
# OnionPIRv2 CUDA Implementation
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (default: 70 for V100, 80 for A100, 89 for L40/4090, 90 for H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# ============================================================================
# Sources
# ============================================================================

set(KERNEL_SOURCES
    src/kernel/ntt.cu
    src/kernel/crt.cu
    src/kernel/elementwise.cu
)

set(CRYPTO_SOURCES
    src/crypto/random.cu
    src/crypto/context.cu
    src/crypto/keygen.cu
    src/crypto/encrypt.cu
)

set(ARITH_SOURCES
    src/arith/decompose.cu
    src/arith/external_product.cu
    src/arith/keyswitch.cu
)

set(EVAL_SOURCES
    src/eval/automorphism.cu
    src/eval/hmux.cu
)

set(MODULE_SOURCES
    src/module/database.cu
    src/module/pir.cu
    src/module/batch.cu
)

set(ALL_SOURCES
    ${KERNEL_SOURCES}
    ${CRYPTO_SOURCES}
    ${ARITH_SOURCES}
    ${EVAL_SOURCES}
    ${MODULE_SOURCES}
)

# ============================================================================
# Library
# ============================================================================

add_library(onionpir STATIC ${ALL_SOURCES})

target_include_directories(onionpir PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(onionpir PUBLIC
    curand
)

# Enable position-independent code for shared library usage
set_target_properties(onionpir PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Benchmark executable
# ============================================================================

add_executable(onionpir_bench src/main.cu)

target_link_libraries(onionpir_bench PRIVATE onionpir)

target_include_directories(onionpir_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(onionpir_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Tests (optional)
# ============================================================================

option(BUILD_TESTS "Build test suite" OFF)

if(BUILD_TESTS)
    add_executable(onionpir_tests tests/test_main.cu)
    target_link_libraries(onionpir_tests PRIVATE onionpir)
    target_include_directories(onionpir_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    set_target_properties(onionpir_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================================
# Install
# ============================================================================

install(TARGETS onionpir
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/onionpir
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

install(FILES include/onionpir.cuh
    DESTINATION include
)

# ============================================================================
# Generate compile_commands.json for IDE support
# ============================================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
