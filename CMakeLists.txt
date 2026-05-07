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
    project(collider LANGUAGES CXX OBJCXX)
elseif(WIN32)
    set(COLLIDER_PLATFORM_WINDOWS TRUE)
    project(collider LANGUAGES CXX CUDA)
else()
    set(COLLIDER_PLATFORM_LINUX TRUE)
    project(collider LANGUAGES CXX CUDA)
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

# Pro feature gating (Wave 6, 2026-05-04; wired up 2026-05-05).
# theCollider Pro adds the brain wallet pipeline. The Free build (default
# when this option is OFF) supports the puzzle / Kangaroo solver and the
# JLP pool client, but rejects --brainwallet at the CLI.
#
# When COLLIDER_PRO=OFF (default):
#   - All brain wallet source files (.cu/.cpp) are excluded from the build.
#   - The COLLIDER_PRO compile definition is NOT set, so #ifdef COLLIDER_PRO
#     blocks in main.cpp / fused_pipeline.cu compile out cleanly.
#   - Pro headers (mega_fused_kernel.hpp, brain_wallet_gpu.hpp, etc.) are not
#     included anywhere in the free build.
#
# When COLLIDER_PRO=ON:
#   - All in-tree brain wallet sources compile and link.
#   - The COLLIDER_PRO compile definition is exposed to every target.
#
# Eventually pro sources will move to a private submodule (hevnsnt/collider-pro)
# at pro/. See docs/PRO-MIGRATION.md.
option(COLLIDER_PRO "Enable Pro brain-wallet pipeline (requires private sources)" OFF)
if(COLLIDER_PRO)
    add_compile_definitions(COLLIDER_PRO=1)
    message(STATUS "Building theCollider PRO (brain wallet enabled)")
else()
    message(STATUS "Building theCollider FREE (brain wallet disabled)")
endif()

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

message(STATUS "TheCollider backend: ${COLLIDER_BACKEND}")

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

find_package(CURL QUIET)
if(CURL_FOUND)
    message(STATUS "libcurl found - license validation enabled")
else()
    message(STATUS "libcurl not found - license validation will be disabled")
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

# Pro-only host-side sources for the brain wallet pipeline.
# Excluded entirely from the free build (no #ifdef wrapping inside the .cpp;
# the file simply is not handed to the compiler).
if(COLLIDER_PRO)
    list(APPEND CORE_SOURCES src/gpu/brain_wallet_gpu.cpp)
endif()

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

    # Windows socket + crypto libs (ws2_32 for sockets, crypt32 for cert store bridge)
    if(WIN32)
        target_link_libraries(collider_pool ws2_32 crypt32)
    endif()

    # OpenSSL for TLS support
    if(OPENSSL_FOUND)
        target_link_libraries(collider_pool OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(collider_pool PUBLIC COLLIDER_HAS_OPENSSL)
    endif()

    message(STATUS "Pool client library enabled")
endif()

# ============================================================================
# Pro License Check Library (Pro build only, requires libcurl + OpenSSL)
# ============================================================================
if(COLLIDER_PRO AND CURL_FOUND AND OPENSSL_FOUND AND
   EXISTS "${CMAKE_SOURCE_DIR}/src/license/license_check.cpp")
    add_library(collider_license STATIC src/license/license_check.cpp)
    target_include_directories(collider_license PUBLIC ${CMAKE_SOURCE_DIR}/src/license)
    target_link_libraries(collider_license
        CURL::libcurl
        OpenSSL::SSL OpenSSL::Crypto
    )
    if(WIN32)
        target_link_libraries(collider_license ws2_32 crypt32)
    endif()
    message(STATUS "License validation library enabled")
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

    # Pro-only GPU sources. The brain-wallet rule engine, mega-fused kernel,
    # and brain-wallet-specific kernels are excluded from the free build at
    # the source-file level (not via #ifdef inside the file). The puzzle /
    # kangaroo / hash160 kernels above are shared with the free build.
    if(COLLIDER_PRO)
        list(APPEND GPU_SOURCES
            src/gpu/gpu_rule_kernel.cu
            src/gpu/gpu_rules.cu
            src/gpu/gpu_rules.cpp
            src/gpu/mega_fused_kernel.cu
        )
    endif()

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
endif()

# ============================================================================
# Main Executable
# ============================================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
    # For CUDA backend, include stub to enable device linking
    if(COLLIDER_BACKEND STREQUAL "CUDA")
        add_executable(collider src/main.cpp src/cuda_stub.cu)
    else()
        add_executable(collider src/main.cpp)
    endif()

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

    if(TARGET collider_license)
        list(APPEND MAIN_LIBS collider_license)
    endif()

    if(COLLIDER_BACKEND STREQUAL "CUDA")
        list(APPEND MAIN_LIBS CUDA::cudart)
    elseif(COLLIDER_BACKEND STREQUAL "METAL")
        list(APPEND MAIN_LIBS ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
    endif()

    target_link_libraries(collider ${MAIN_LIBS})

    # For CUDA separable compilation, the final executable must resolve device symbols
    if(COLLIDER_BACKEND STREQUAL "CUDA")
        set_target_properties(collider PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_RUNTIME_LIBRARY Shared
        )
    endif()
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

    # CLI parser + config-override matrix (Wave 5 / track-e, 2026-05-04).
    # Header-only test against src/core/yaml_config.hpp; mirrors the parse_args
    # in src/main.cpp (kept in sync; if you add a CLI flag, mirror it here).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_cli_parser.cpp")
        add_executable(test_cli_parser tests/test_cli_parser.cpp)
        target_link_libraries(test_cli_parser Threads::Threads)
        add_test(NAME CLIParser COMMAND test_cli_parser)
    endif()

    # Hashcat rule engine known-answer test. Pins the engine to externally
    # documented rule semantics so a regression vs. hashcat behaviour fails
    # loudly. Header-only, runs everywhere (no GPU / network required).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_rule_engine_kat.cpp" AND TARGET collider_core)
        add_executable(test_rule_engine_kat tests/test_rule_engine_kat.cpp)
        target_link_libraries(test_rule_engine_kat collider_core)
        add_test(NAME RuleEngineKAT COMMAND test_rule_engine_kat)
    endif()

    # JLP pool TLS handshake live integration test. Connects to
    # collisionprotocol.com:17403, runs a verified TLS handshake, sends an
    # AUTH frame, reads the response. Returns 77 (skip) if
    # COLLIDER_SKIP_NETWORK_TESTS is set or TCP connect fails within 5
    # seconds. Cert/protocol failures are real FAILs.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_handshake.cpp" AND OPENSSL_FOUND)
        add_executable(test_jlp_pool_handshake tests/test_jlp_pool_handshake.cpp)
        target_link_libraries(test_jlp_pool_handshake
            OpenSSL::SSL OpenSSL::Crypto Threads::Threads
        )
        if(WIN32)
            # ws2_32: WinSock for the test's own TCP/select code.
            # crypt32: CertOpenSystemStoreA / CertEnumCertificatesInStore /
            #          CertCloseStore. Required because vcpkg OpenSSL on
            #          Windows has no usable default trust store; the test
            #          bridges the Windows ROOT cert store into OpenSSL's
            #          X509_STORE so chain verification works.
            target_link_libraries(test_jlp_pool_handshake ws2_32 crypt32)
        endif()
        add_test(NAME JLPPoolHandshake COMMAND test_jlp_pool_handshake)
        set_tests_properties(JLPPoolHandshake PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # JLP pool URL parser + DistinguishedPoint serialization unit tests.
    # No sockets, no threads -- pure logic against the collider_pool library.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_manager.cpp" AND TARGET collider_pool)
        add_executable(test_jlp_pool_manager tests/test_jlp_pool_manager.cpp)
        target_link_libraries(test_jlp_pool_manager collider_pool Threads::Threads)
        add_test(NAME JLPPoolManager COMMAND test_jlp_pool_manager)
    endif()

    # JLP pool wire-protocol tests against an in-process mock TCP server on
    # 127.0.0.1 (use_tls=false). Drives the AUTH state machine, work
    # assignment, stats, DP submission, ping/pong, and pre-auth gating.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_protocol.cpp" AND TARGET collider_pool)
        add_executable(test_jlp_pool_protocol tests/test_jlp_pool_protocol.cpp)
        target_link_libraries(test_jlp_pool_protocol collider_pool Threads::Threads)
        if(WIN32)
            # ws2_32 for the test's own mock-server WinSock calls (the
            # collider_pool library transitively links it too, but listing it
            # here keeps the dependency explicit).
            target_link_libraries(test_jlp_pool_protocol ws2_32)
        endif()
        add_test(NAME JLPPoolProtocol COMMAND test_jlp_pool_protocol)
        # The protocol suite uses real threads + sockets; give it room.
        set_tests_properties(JLPPoolProtocol PROPERTIES TIMEOUT 60)
    endif()

    # JLP pool reconnect / disconnect handling tests. Mock TCP server, no TLS.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_reconnect.cpp" AND TARGET collider_pool)
        add_executable(test_jlp_pool_reconnect tests/test_jlp_pool_reconnect.cpp)
        target_link_libraries(test_jlp_pool_reconnect collider_pool Threads::Threads)
        if(WIN32)
            target_link_libraries(test_jlp_pool_reconnect ws2_32)
        endif()
        add_test(NAME JLPPoolReconnect COMMAND test_jlp_pool_reconnect)
        set_tests_properties(JLPPoolReconnect PROPERTIES TIMEOUT 60)
    endif()

    # CUDA-only GPU correctness tests (Wave 0, 2026-05-04 review).
    # Each test is expected to FAIL on current code and PASS after Wave 1 crypto fixes.
    # Tests return CTest skip code (77) when no CUDA device is present at runtime.
    if(COLLIDER_BACKEND STREQUAL "CUDA" AND TARGET collider_gpu)
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_secp256k1_inv.cu")
            add_executable(test_secp256k1_inv tests/test_secp256k1_inv.cu)
            set_target_properties(test_secp256k1_inv PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_secp256k1_inv collider_gpu CUDA::cudart)
            add_test(NAME Secp256k1Inv COMMAND test_secp256k1_inv)
            set_tests_properties(Secp256k1Inv PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_ec_table_consistency.cu")
            add_executable(test_ec_table_consistency tests/test_ec_table_consistency.cu)
            set_target_properties(test_ec_table_consistency PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_ec_table_consistency collider_gpu CUDA::cudart)
            add_test(NAME EcTableConsistency COMMAND test_ec_table_consistency)
            set_tests_properties(EcTableConsistency PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        # Wave 1 Track C C-CRIT-2 follow-up (2026-05-04). Validates the
        # binary-exponentiation mod_inv in puzzle_optimized.cu (separate copy
        # from secp256k1.cu's mod_inv because U256 here is 4 x uint64).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_puzzle_optimized_inv.cu")
            add_executable(test_puzzle_optimized_inv tests/test_puzzle_optimized_inv.cu)
            set_target_properties(test_puzzle_optimized_inv PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_puzzle_optimized_inv collider_gpu CUDA::cudart)
            add_test(NAME PuzzleOptimizedInv COMMAND test_puzzle_optimized_inv)
            set_tests_properties(PuzzleOptimizedInv PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        # GpuHash160 exercises the fused brain-wallet pipeline
        # (fused_pipeline_init + fused_brain_wallet_batch_fixed_stride) and is
        # therefore a Pro-only test. In the free build the brain-wallet
        # kernels and host wrappers do not exist in the binary, so the test
        # cannot link.
        if(COLLIDER_PRO AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_gpu_hash160.cu" AND TARGET collider_core)
            add_executable(test_gpu_hash160 tests/test_gpu_hash160.cu)
            set_target_properties(test_gpu_hash160 PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_gpu_hash160 collider_gpu collider_core CUDA::cudart)
            add_test(NAME GpuHash160 COMMAND test_gpu_hash160)
            set_tests_properties(GpuHash160 PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        # EC scalar multiplication known-answer test. Hands the GPU a few well
        # known privkeys (1, 2, 3, 7), reads back the affine pubkey, encodes
        # it as the standard 33-byte compressed pubkey, and compares against
        # the canonical secp256k1 generator multiples. Catches regressions in
        # the entire EC pipeline (mod_mul, mod_inv, ec_double, ec_add_mixed,
        # jacobian_to_affine, precomputed-table generation).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_ec_mul_known_answers.cu")
            add_executable(test_ec_mul_known_answers tests/test_ec_mul_known_answers.cu)
            set_target_properties(test_ec_mul_known_answers PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_ec_mul_known_answers collider_gpu CUDA::cudart)
            add_test(NAME EcMulKnownAnswers COMMAND test_ec_mul_known_answers)
            set_tests_properties(EcMulKnownAnswers PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        # Known-answer test for the puzzle / kangaroo EC multiplication path
        # (puzzle_optimized.cu's ec_mul_glv). Calls a dedicated test kernel
        # that runs ec_mul_glv on a handful of small scalars and writes the
        # affine pubkey back to host; the host re-encodes as compressed
        # pubkey and byte-compares against known answers (same vectors as
        # EcMulKnownAnswers, but exercising the puzzle_optimized.cu code path
        # rather than secp256k1.cu). Hash chain coverage lives in GpuHash160
        # and test_hash_vectors; splitting the concerns gives actionable
        # failure signals.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_kangaroo_small_puzzle.cu")
            add_executable(test_kangaroo_small_puzzle tests/test_kangaroo_small_puzzle.cu)
            set_target_properties(test_kangaroo_small_puzzle PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_kangaroo_small_puzzle collider_gpu CUDA::cudart)
            add_test(NAME KangarooSmallPuzzle COMMAND test_kangaroo_small_puzzle)
            set_tests_properties(KangarooSmallPuzzle PROPERTIES SKIP_RETURN_CODE 77)
        endif()
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
message(STATUS "TheCollider Configuration Summary:")
message(STATUS "  Platform:    ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Backend:     ${COLLIDER_BACKEND}")
message(STATUS "  Build Type:  ${CMAKE_BUILD_TYPE}")
if(COLLIDER_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA Archs:  ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Tests:       ${COLLIDER_BUILD_TESTS}")
message(STATUS "  Benchmarks:  ${COLLIDER_BUILD_BENCHMARKS}")
message(STATUS "  Tools:       ${COLLIDER_BUILD_TOOLS}")
message(STATUS "")
