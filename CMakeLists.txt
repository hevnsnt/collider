cmake_minimum_required(VERSION 3.20)

# ============================================================================
# Cross-Platform GPU Compute Project
# Supports: CUDA (Windows/Linux), Metal (macOS), CPU fallback
# ============================================================================

# ============================================================================
# vcpkg Auto-Bootstrap (Windows only - downloads dependencies automatically)
# ============================================================================
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Check if vcpkg is already installed
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
        message(STATUS "Using existing vcpkg: $ENV{VCPKG_ROOT}")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
        message(STATUS "Using local vcpkg")
    else()
        # Auto-download vcpkg
        message(STATUS "Downloading vcpkg for automatic dependency management...")
        include(FetchContent)
        FetchContent_Declare(
            vcpkg
            GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
            GIT_TAG 2024.01.12
            GIT_SHALLOW TRUE
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/vcpkg
        )
        FetchContent_MakeAvailable(vcpkg)
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

        # Bootstrap vcpkg
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/vcpkg.exe")
            message(STATUS "Bootstrapping vcpkg...")
            execute_process(
                COMMAND cmd /c "${CMAKE_SOURCE_DIR}/vcpkg/bootstrap-vcpkg.bat" -disableMetrics
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/vcpkg
                RESULT_VARIABLE VCPKG_BOOTSTRAP_RESULT
            )
            if(NOT VCPKG_BOOTSTRAP_RESULT EQUAL 0)
                message(WARNING "vcpkg bootstrap failed - OpenSSL may not be available")
            endif()
        endif()
    endif()

    # Set triplet for x64 Windows
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "")
endif()

# Detect platform before project declaration
if(APPLE)
    set(COLLIDER_PLATFORM_MACOS TRUE)
    project(collider LANGUAGES C CXX OBJCXX)
elseif(WIN32)
    set(COLLIDER_PLATFORM_WINDOWS TRUE)
    project(collider LANGUAGES C CXX CUDA)
else()
    set(COLLIDER_PLATFORM_LINUX TRUE)
    project(collider LANGUAGES C CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Options
# ============================================================================
option(COLLIDER_USE_CUDA "Enable CUDA backend" ON)
option(COLLIDER_USE_METAL "Enable Metal backend" ON)
option(COLLIDER_USE_CPU "Enable CPU fallback backend" ON)
option(COLLIDER_BUILD_TESTS "Build tests" ON)
option(COLLIDER_BUILD_BENCHMARKS "Build benchmarks" ON)
option(COLLIDER_BUILD_TOOLS "Build CLI tools" ON)

# ============================================================================
# Backend Detection
# ============================================================================
set(COLLIDER_BACKEND "CPU")  # Default

if(APPLE AND COLLIDER_USE_METAL)
    # macOS: Use Metal
    set(COLLIDER_BACKEND "METAL")
    add_compile_definitions(COLLIDER_USE_METAL=1)
    add_compile_definitions(COLLIDER_PLATFORM_MACOS=1)

    # Check for Apple Silicon
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(ARCH STREQUAL "arm64")
        add_compile_definitions(COLLIDER_APPLE_SILICON=1)
        message(STATUS "Detected Apple Silicon (${ARCH})")
    endif()

elseif(COLLIDER_USE_CUDA)
    # Windows/Linux: Try CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(COLLIDER_BACKEND "CUDA")
        enable_language(CUDA)
        add_compile_definitions(COLLIDER_USE_CUDA=1)

        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # Detect GPU architecture
        # RTX 5090 (Blackwell): SM 10.0
        # RTX 4090 (Ada): SM 8.9
        # RTX 3060/3070/3080/3090 (Ampere): SM 8.6
        # We compile for all to support various cards
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Check if we can detect the GPU
            set(CMAKE_CUDA_ARCHITECTURES "86;89;100")
            message(STATUS "Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()

        # Force consistent CUDA runtime library (avoid mixing static/dynamic)
        # Use Shared (cudart.lib) to avoid duplicate symbol errors on Windows
        set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

        # CUDA optimization flags - aggressive optimization for compute-bound kernels
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
        # Additional CUDA optimizations for high-performance computing
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --extra-device-vectorization")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --restrict")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --fmad=true")
        # Lineinfo for profiling (minimal overhead, enables nsys/ncu profiling)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
        # Suppress deprecated GPU architecture warnings (for older GPUs like GTX 9xx)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Wno-deprecated-gpu-targets")

        # Debug/Profile build for detailed profiling
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -G -lineinfo --use_fast_math")

        if(WIN32)
            add_compile_definitions(COLLIDER_PLATFORM_WINDOWS=1)
            add_compile_definitions(NOMINMAX)           # Prevent Windows min/max macro conflicts
            add_compile_definitions(WIN32_LEAN_AND_MEAN) # Reduce Windows header bloat
        else()
            add_compile_definitions(COLLIDER_PLATFORM_LINUX=1)
        endif()
    endif()
endif()

if(COLLIDER_BACKEND STREQUAL "CPU")
    add_compile_definitions(COLLIDER_USE_CPU=1)
    message(STATUS "Using CPU fallback backend")
endif()

message(STATUS "Collider backend: ${COLLIDER_BACKEND}")

# ============================================================================
# Edition Configuration  
# ============================================================================
    message(STATUS "Building Collider Free edition (pool-only)")

# ============================================================================
# Optimization Flags
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
        if(ARCH STREQUAL "arm64")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=apple-m1")
        endif()
    elseif(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2 /GL /EHsc")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    endif()
endif()

# ============================================================================
# Link-Time Optimization (LTO/IPO) - 5-15% performance boost
# ============================================================================
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "LTO/IPO enabled for Release builds")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "LTO/IPO disabled (not Release build)")
    else()
        message(WARNING "LTO/IPO not supported: ${IPO_ERROR}")
    endif()
endif()

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(Threads REQUIRED)

# OpenSSL - help CMake find OpenSSL on various platforms
if(APPLE)
    # Apple Silicon Homebrew path
    if(EXISTS "/opt/homebrew/opt/openssl")
        set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl")
    # Intel Mac Homebrew path
    elseif(EXISTS "/usr/local/opt/openssl")
        set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl")
    endif()
elseif(WIN32)
    # Common Windows OpenSSL installation paths
    set(_WIN_OPENSSL_PATHS
        "C:/Program Files/OpenSSL-Win64"
        "C:/Program Files/OpenSSL"
        "C:/OpenSSL-Win64"
        "C:/OpenSSL"
        "$ENV{VCPKG_ROOT}/installed/x64-windows"
    )
    foreach(_PATH ${_WIN_OPENSSL_PATHS})
        if(EXISTS "${_PATH}")
            set(OPENSSL_ROOT_DIR "${_PATH}")
            break()
        endif()
    endforeach()
endif()

find_package(OpenSSL QUIET)
if(OPENSSL_FOUND)
    message(STATUS "OpenSSL found - TLS support enabled")
    message(STATUS "  OpenSSL version: ${OPENSSL_VERSION}")
    message(STATUS "  OpenSSL include: ${OPENSSL_INCLUDE_DIR}")
else()
    message(STATUS "OpenSSL not found - TLS support disabled")
endif()

# XXH3 - SIMD-optimized hash for bloom filter (header-only)
include(FetchContent)
FetchContent_Declare(
    xxhash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG v0.8.2
)
FetchContent_MakeAvailable(xxhash)
# xxHash is header-only, just need include path
set(XXHASH_INCLUDE_DIR ${xxhash_SOURCE_DIR})

if(APPLE AND COLLIDER_BACKEND STREQUAL "METAL")
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(COCOA_LIBRARY Cocoa)
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/src)

# ============================================================================
# Platform Abstraction Library
# ============================================================================
set(PLATFORM_SOURCES
    src/platform/platform.hpp
)

if(COLLIDER_BACKEND STREQUAL "CUDA")
    list(APPEND PLATFORM_SOURCES src/platform/cuda_platform.cu)
elseif(COLLIDER_BACKEND STREQUAL "METAL")
    list(APPEND PLATFORM_SOURCES src/platform/metal_platform.mm)
else()
    list(APPEND PLATFORM_SOURCES src/platform/cpu_platform.cpp)
endif()

if(COLLIDER_BACKEND STREQUAL "METAL")
    add_library(collider_platform STATIC ${PLATFORM_SOURCES})
    target_link_libraries(collider_platform
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
        Threads::Threads
    )
    set_source_files_properties(src/platform/metal_platform.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
elseif(COLLIDER_BACKEND STREQUAL "CUDA")
    add_library(collider_platform STATIC ${PLATFORM_SOURCES})
    target_link_libraries(collider_platform
        CUDA::cudart
        Threads::Threads
    )
    set_target_properties(collider_platform PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(collider_platform STATIC ${PLATFORM_SOURCES})
    target_link_libraries(collider_platform
        Threads::Threads
    )
endif()

# ============================================================================
# Core Library (CPU-side intelligence layer)
# ============================================================================
set(CORE_SOURCES
    src/core/rule_engine.cpp
    src/generators/passphrase_generator.cpp
    src/generators/priority_queue.cpp
)

# Check if source files exist
foreach(SRC ${CORE_SOURCES})
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${SRC}")
        list(REMOVE_ITEM CORE_SOURCES ${SRC})
    endif()
endforeach()

if(CORE_SOURCES)
    add_library(collider_core STATIC ${CORE_SOURCES})
    target_link_libraries(collider_core
        collider_platform
        Threads::Threads
    )
    target_include_directories(collider_core PUBLIC ${XXHASH_INCLUDE_DIR})
    
else()
    # Create empty library if no sources exist yet
    add_library(collider_core INTERFACE)
    target_link_libraries(collider_core INTERFACE
        collider_platform
        Threads::Threads
    )
    target_include_directories(collider_core INTERFACE ${XXHASH_INCLUDE_DIR})
    
endif()

# ============================================================================
# Pool Client Library (Network-based distributed solving)
# ============================================================================
set(POOL_SOURCES
    src/pool/jlp_pool_client.cpp
    src/pool/http_pool_client.cpp
    src/pool/pool_manager.cpp
)

# Check if source files exist
set(EXISTING_POOL_SOURCES)
foreach(SRC ${POOL_SOURCES})
    if(EXISTS "${CMAKE_SOURCE_DIR}/${SRC}")
        list(APPEND EXISTING_POOL_SOURCES ${SRC})
    endif()
endforeach()

if(EXISTING_POOL_SOURCES)
    add_library(collider_pool STATIC ${EXISTING_POOL_SOURCES})
    target_link_libraries(collider_pool
        Threads::Threads
    )
    target_include_directories(collider_pool PUBLIC
        ${CMAKE_SOURCE_DIR}/src/pool
    )

    # Windows socket library
    if(WIN32)
        target_link_libraries(collider_pool ws2_32)
    endif()

    # OpenSSL for TLS support
    if(OPENSSL_FOUND)
        target_link_libraries(collider_pool OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(collider_pool PUBLIC COLLIDER_HAS_OPENSSL)
    endif()

    message(STATUS "Pool client library enabled")
endif()

# ============================================================================
# RCKangaroo - High-performance Kangaroo solver (GPLv3, RetiredCoder)
# ============================================================================
if(COLLIDER_BACKEND STREQUAL "CUDA")
    option(COLLIDER_USE_RCKANGAROO "Use RCKangaroo as Kangaroo solver backend" ON)
    if(COLLIDER_USE_RCKANGAROO AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/RCKangaroo")
        add_subdirectory(third_party/RCKangaroo)
        add_compile_definitions(COLLIDER_USE_RCKANGAROO=1)
        message(STATUS "RCKangaroo enabled - high-performance Kangaroo solver")
    endif()
endif()

# ============================================================================
# GPU Kernels
# ============================================================================
if(COLLIDER_BACKEND STREQUAL "CUDA")
    set(GPU_SOURCES
        src/gpu/sha256.cu
        src/gpu/secp256k1.cu
        src/gpu/ripemd160.cu
        src/gpu/bloom_filter.cu
        src/gpu/h160_bloom_filter.cu
        src/gpu/pipeline.cu
        src/gpu/fused_pipeline.cu
        src/gpu/puzzle_gpu.cu
        src/gpu/puzzle_optimized.cu
        src/gpu/kangaroo_kernel.cu
    )

    # Add RCKangaroo wrapper if enabled
    if(COLLIDER_USE_RCKANGAROO AND TARGET rckangaroo)
        list(APPEND GPU_SOURCES src/gpu/rckangaroo_wrapper.cu)
    endif()

    # Check if source files exist
    set(EXISTING_GPU_SOURCES)
    foreach(SRC ${GPU_SOURCES})
        if(EXISTS "${CMAKE_SOURCE_DIR}/${SRC}")
            list(APPEND EXISTING_GPU_SOURCES ${SRC})
        endif()
    endforeach()

    if(EXISTING_GPU_SOURCES)
        add_library(collider_gpu STATIC ${EXISTING_GPU_SOURCES})
        target_link_libraries(collider_gpu
            CUDA::cudart
        )
        # Link RCKangaroo if enabled
        if(COLLIDER_USE_RCKANGAROO AND TARGET rckangaroo)
            target_link_libraries(collider_gpu rckangaroo)
            target_include_directories(collider_gpu PRIVATE
                ${CMAKE_SOURCE_DIR}/third_party/RCKangaroo
            )
        endif()
        set_target_properties(collider_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY Shared
            POSITION_INDEPENDENT_CODE ON
        )
    endif()

elseif(COLLIDER_BACKEND STREQUAL "METAL")
    # Metal shaders are compiled separately
    # We use .metal files compiled to .metallib at build time
    set(METAL_SHADERS
        src/gpu/sha256.metal
        src/gpu/secp256k1.metal
        src/gpu/ripemd160.metal
        src/gpu/bloom_filter.metal
    )

    # For now, create placeholder library
    # Metal compute kernels will be loaded at runtime
    add_library(collider_gpu INTERFACE)

else()
    # CPU fallback - provide stub implementations
    add_library(collider_gpu INTERFACE)
endif()

# ============================================================================
# Tools
# ============================================================================
if(COLLIDER_BUILD_TOOLS)
    # UTXO Bloom Filter Builder
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/build_bloom.cpp")
        add_executable(build_bloom
            src/tools/build_bloom.cpp
        )
        if(TARGET collider_core)
            target_link_libraries(build_bloom
                collider_core
                collider_platform
                Threads::Threads
            )
        else()
            target_link_libraries(build_bloom
                collider_platform
                Threads::Threads
            )
        endif()
    endif()

    # Banner Test Tool
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/test_banner.cpp")
        add_executable(test_banner
            src/tools/test_banner.cpp
        )
        target_link_libraries(test_banner
            Threads::Threads
        )
    endif()

    # License Generator (Pro Edition Only - Private Tool)
endif()

# ============================================================================
# Main Executables
# ============================================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
    set(MAIN_LIBS collider_platform Threads::Threads)

    if(TARGET collider_core)
        list(APPEND MAIN_LIBS collider_core)
    endif()

    if(TARGET collider_gpu)
        list(APPEND MAIN_LIBS collider_gpu)
    endif()

    if(TARGET collider_pool)
        list(APPEND MAIN_LIBS collider_pool)
    endif()

    if(COLLIDER_BACKEND STREQUAL "CUDA")
        list(APPEND MAIN_LIBS CUDA::cudart)
    elseif(COLLIDER_BACKEND STREQUAL "METAL")
        list(APPEND MAIN_LIBS ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
    endif()

    # ========================================================================
    # Shared Sources Lists
    # ========================================================================
    set(SHARED_SOURCES
        src/main.cpp
    )
    
    if(COLLIDER_BACKEND STREQUAL "CUDA")
        list(APPEND SHARED_SOURCES src/cuda_stub.cu)
    endif()

    # ========================================================================
    # Free Edition Executable
    # ========================================================================
    add_executable(collider ${SHARED_SOURCES})
    target_compile_definitions(collider PRIVATE COLLIDER_FREE=1)
    target_link_libraries(collider ${MAIN_LIBS})
    
    if(COLLIDER_BACKEND STREQUAL "CUDA")
        set_target_properties(collider PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_RUNTIME_LIBRARY Shared
        )
    endif()

    # ============================================================================
# Tests
# ============================================================================
if(COLLIDER_BUILD_TESTS)
    enable_testing()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_rule_engine.cpp" AND TARGET collider_core)
        add_executable(test_rule_engine tests/test_rule_engine.cpp)
        target_link_libraries(test_rule_engine collider_core)
        add_test(NAME RuleEngine COMMAND test_rule_engine)
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_priority_queue.cpp" AND TARGET collider_core)
        add_executable(test_priority_queue tests/test_priority_queue.cpp)
        target_link_libraries(test_priority_queue collider_core)
        add_test(NAME PriorityQueue COMMAND test_priority_queue)
    endif()

    # Platform tests
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_platform.cpp")
        add_executable(test_platform tests/test_platform.cpp)
        target_link_libraries(test_platform collider_platform Threads::Threads)
        if(COLLIDER_BACKEND STREQUAL "CUDA")
            target_link_libraries(test_platform CUDA::cudart)
        elseif(COLLIDER_BACKEND STREQUAL "METAL")
            target_link_libraries(test_platform ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
        endif()
        add_test(NAME Platform COMMAND test_platform)
    endif()

    # Hash function verification tests
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_hash_vectors.cpp")
        add_executable(test_hash_vectors tests/test_hash_vectors.cpp)
        target_link_libraries(test_hash_vectors Threads::Threads)
        add_test(NAME HashVectors COMMAND test_hash_vectors)
    endif()
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(COLLIDER_BUILD_BENCHMARKS AND COLLIDER_BACKEND STREQUAL "CUDA")
    if(EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_gpu_pipeline.cu")
        add_executable(bench_gpu_pipeline benchmarks/bench_gpu_pipeline.cu)
        target_link_libraries(bench_gpu_pipeline
            collider_gpu
            CUDA::cudart
        )
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================
if(TARGET collider)
    install(TARGETS collider RUNTIME DESTINATION bin)
endif()

endif()

if(TARGET build_bloom)
    install(TARGETS build_bloom RUNTIME DESTINATION bin)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/rules")
    install(DIRECTORY rules/ DESTINATION share/collider/rules)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/wordlists")
    install(DIRECTORY wordlists/ DESTINATION share/collider/wordlists)
endif()


# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "Collider Configuration Summary:")
message(STATUS "  Platform:    ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Backend:     ${COLLIDER_BACKEND}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Edition:     Free (pool-only)")
if(COLLIDER_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA Archs:  ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Tests:       ${COLLIDER_BUILD_TESTS}")
message(STATUS "  Benchmarks:  ${COLLIDER_BUILD_BENCHMARKS}")
message(STATUS "  Tools:       ${COLLIDER_BUILD_TOOLS}")
    message(STATUS "")
    message(STATUS "Executable built:")
    message(STATUS "  collider     - Free edition (pool-only)")
message(STATUS "")
