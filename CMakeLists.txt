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
        # Phase 5+ kernels (v1.4.0) use lambdas + relaxed-constexpr inside __device__
        # functions. Apply to all configurations, not just Release.
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

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

if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release"
   AND NOT COLLIDER_DISABLE_IPO)
    message(STATUS "LTO/IPO enabled for Release builds (host C/C++ only)")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    # CUDA device LTO (-dlto) pushes nvlink past its 1MB default stack on
    # Windows for the collider_pro symbol graph.
    set(CMAKE_CUDA_INTERPROCEDURAL_OPTIMIZATION FALSE)
    message(STATUS "LTO/IPO disabled for CUDA device link (avoids nvlink stack overflow)")
else()
    if(COLLIDER_DISABLE_IPO)
        message(STATUS "LTO/IPO disabled by COLLIDER_DISABLE_IPO=ON")
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
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
    src/core/cpu_kangaroo_backend.cpp
    src/core/kangaroo_backend_factory.cpp
    src/generators/passphrase_generator.cpp
    src/generators/priority_queue.cpp
    # v1.4.1 A.3: CLI argument parsing extracted from main.cpp so the
    # runtime drivers and the tests/test_cli_parser target can share a
    # single source of truth for Arguments / parse_args / print_usage.
    src/cli/cli_parser.cpp
    # v1.4.1 A.3: top-level interactive flow (banner + main menu + per-mode
    # submenus + Windows ANSI enabler) extracted from main.cpp. The brain
    # wallet submenu inside is gated on COLLIDER_PRO at the call-site, but
    # the file itself compiles in both editions.
    src/ui/interactive_ui.cpp
    # v1.4.1 A.3: pool-mode runtime driver extracted from main.cpp. Free +
    # Pro both build it; the bloom-filter feature inside is #ifdef-gated.
    src/runtime/pool_solver.cpp
    # v1.4.1 A.3 (5/6): puzzle-mode + benchmark runtime drivers extracted
    # from main.cpp. Available in BOTH Free and Pro builds; the brain-wallet
    # benchmark variant inside is gated by #ifdef COLLIDER_PRO at the call
    # site (Pro-only includes from gpu/brain_wallet_gpu.hpp). The Free path
    # uses the CPU + Metal/CUDA SHA-256 benchmark with no Pro symbols.
    src/runtime/puzzle_solver.cpp
    # v1.4.1 A.3 (6/6): GPU detection helper extracted from main.cpp so
    # the dispatcher (main.cpp) and the runtime drivers can both link
    # against detect_gpus() without main.cpp owning the body.
    src/runtime/gpu_detection.cpp
    # v1.4.1 UI: live BTC balance lookup against mempool.space for the
    # solved-puzzle banner. libcurl-gated -- if CURL_FOUND was false at
    # configure time, the function returns nullopt and the banner shows
    # "balance unavailable (offline)".
    src/ui/btc_balance.cpp
)

# Free-licensed Metal SHA-256 benchmark (Mac only). Available in BOTH
# Free and Pro builds so `--benchmark` can validate the GPU on Metal.
if(APPLE AND COLLIDER_USE_METAL
   AND EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/sha256_metal_bench.mm")
    list(APPEND CORE_SOURCES src/gpu/sha256_metal_bench.mm)
    set_source_files_properties(src/gpu/sha256_metal_bench.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# Metal Kangaroo solver (Phase 1 of Mac pool support, v1.4.0). Free-
# licensed; available in BOTH Free and Pro Mac builds so pool mining
# on Apple Silicon stops falling back to CPU kangaroo at ~200 K/s.
if(APPLE AND COLLIDER_USE_METAL
   AND EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/kangaroo_metal.mm")
    list(APPEND CORE_SOURCES src/gpu/kangaroo_metal.mm)
    set_source_files_properties(src/gpu/kangaroo_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    # Metal IKangarooBackend adapter -- gated to the same conditions as
    # the underlying KangarooMetalSolver above.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/metal_kangaroo_backend.mm")
        list(APPEND CORE_SOURCES src/gpu/metal_kangaroo_backend.mm)
        set_source_files_properties(src/gpu/metal_kangaroo_backend.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    endif()
    # v1.4.1: Metal-backed MultiGPUKangarooManager for standalone puzzle
    # solving. Pre-1.4.1 the standalone path's MultiGPU class was a
    # no-CUDA stub on Mac that returned false from init() and silently
    # fell back to CPU kangaroo. This file implements the full surface
    # on top of KangarooMetalSolver with host-side collision detection.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/metal_multi_gpu_kangaroo.mm")
        list(APPEND CORE_SOURCES src/gpu/metal_multi_gpu_kangaroo.mm)
        set_source_files_properties(src/gpu/metal_multi_gpu_kangaroo.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    endif()
    # v1.4.1: Metal-backed brute-force puzzle solver. Pre-1.4.1 the Mac
    # build of `collider --puzzle N` (no --kangaroo) hit the no-CUDA stub
    # of MultiGPUPuzzleSolver and silently fell back to the CPU reference
    # at ~30 KKeys/s. puzzle_metal.mm is the per-device dispatcher;
    # metal_multi_gpu_puzzle.mm bridges it to the MultiGPUPuzzleSolver
    # surface that puzzle_solver.cpp consumes.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/puzzle_metal.mm")
        list(APPEND CORE_SOURCES src/gpu/puzzle_metal.mm)
        set_source_files_properties(src/gpu/puzzle_metal.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/metal_multi_gpu_puzzle.mm")
        list(APPEND CORE_SOURCES src/gpu/metal_multi_gpu_puzzle.mm)
        set_source_files_properties(src/gpu/metal_multi_gpu_puzzle.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    endif()
    # Bake the MSL source into a generated C++ header so neither the
    # production dispatcher nor the test binaries need to search the
    # filesystem for kangaroo.metal at runtime. The previous approach
    # (configure_file COPYONLY + dual-loader candidate-path lists in
    # kangaroo_metal.mm and tests/test_metal_secp256k1.mm) had three
    # problems: the candidate lists drifted; running the binary from
    # an unexpected cwd produced "shader not found"; and the staged
    # build/ copy could go stale on parallel builds.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/kangaroo.metal")
        set(_kangaroo_metal_src "${CMAKE_SOURCE_DIR}/src/gpu/kangaroo.metal")
        set(_kangaroo_metal_hdr "${CMAKE_BINARY_DIR}/generated/kangaroo_metal_source.h")
        file(READ "${_kangaroo_metal_src}" KANGAROO_METAL_SOURCE_BODY)
        # Hand the source to the compiler as a raw string literal so
        # quotes / backslashes / newlines pass through untouched. The
        # delimiter `KANGAROO_METAL` must not appear in the .metal text
        # itself; trivially true today.
        configure_file(
            "${CMAKE_SOURCE_DIR}/cmake/kangaroo_metal_source.h.in"
            "${_kangaroo_metal_hdr}"
            @ONLY)
        # Re-run the configure step whenever the .metal file changes so
        # the generated header stays in sync. CMake doesn't track
        # `file(READ)` inputs automatically.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${_kangaroo_metal_src}")
        # Expose the generated/ directory to anything that needs to
        # `#include "kangaroo_metal_source.h"`. Production code lives in
        # collider_core (added below); the test target adds the same
        # directory directly in its own block.
        set(KANGAROO_METAL_GENERATED_INCLUDE_DIR
            "${CMAKE_BINARY_DIR}/generated"
            CACHE INTERNAL "Generated kangaroo MSL header dir")
    endif()
    # Same embedding pattern for the puzzle MSL source. The .h.in template
    # at cmake/puzzle_metal_source.h.in is a near-clone of the kangaroo
    # one; intentionally separate so a future refactor can change one
    # without touching the other.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/puzzle.metal")
        set(_puzzle_metal_src "${CMAKE_SOURCE_DIR}/src/gpu/puzzle.metal")
        set(_puzzle_metal_hdr "${CMAKE_BINARY_DIR}/generated/puzzle_metal_source.h")
        file(READ "${_puzzle_metal_src}" PUZZLE_METAL_SOURCE_BODY)
        configure_file(
            "${CMAKE_SOURCE_DIR}/cmake/puzzle_metal_source.h.in"
            "${_puzzle_metal_hdr}"
            @ONLY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${_puzzle_metal_src}")
    endif()
endif()

# Pro-only host-side sources for the brain wallet pipeline.
# Excluded entirely from the free build (no #ifdef wrapping inside the .cpp;
# the file simply is not handed to the compiler).
if(COLLIDER_PRO)
    list(APPEND CORE_SOURCES src/gpu/brain_wallet_gpu.cpp)
    # v1.4.1 C.1/C.2: pimpl bodies for the brain-wallet generators. Both
    # headers are #include'd from main.cpp only inside its COLLIDER_PRO
    # block; the .cpp companions match that gating so the free build
    # never compiles them.
    list(APPEND CORE_SOURCES src/generators/streaming_brain_wallet.cpp)
    list(APPEND CORE_SOURCES src/generators/brain_wallet_engine.cpp)
    # v1.4.1 A.3: brain wallet runtime driver extracted from main.cpp.
    # Whole TU is COLLIDER_PRO-gated; the file itself wraps everything
    # in #ifdef COLLIDER_PRO defense-in-depth.
    list(APPEND CORE_SOURCES src/runtime/brain_wallet_runner.cpp)
    # v1.4.1 A.3 (6/6): pre-dispatch license activation + startup
    # validation. PRO-only because the underlying license/license_check.hpp
    # is PRO-only; the .cpp wraps everything in #ifdef COLLIDER_PRO too.
    list(APPEND CORE_SOURCES src/runtime/license_gate.cpp)
    # gpu_rules.cpp is fully #ifdef-guarded; on non-CUDA backends it
    # provides stub implementations of GPURuleEngine that return false /
    # 0 / "". Without it, main.cpp's references to GPURuleEngine fail to
    # link on Metal/CPU backends. The CUDA-only .cu kernels remain in
    # GPU_SOURCES under the COLLIDER_BACKEND==CUDA branch.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/gpu_rules.cpp"
       AND NOT COLLIDER_BACKEND STREQUAL "CUDA")
        list(APPEND CORE_SOURCES src/gpu/gpu_rules.cpp)
    endif()
    # Brain Wallet v2 host orchestrator (Phase 9): pure C++, no CUDA TU.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/v2/v2_orchestrator.cpp")
        list(APPEND CORE_SOURCES src/gpu/v2/v2_orchestrator.cpp)
    endif()
    # Phase 11 (v1.4.0): Metal dispatch (Obj-C++). Built only on macOS;
    # the file is otherwise ignored (its symbols are referenced from
    # v2_orchestrator.cpp behind __APPLE__ + COLLIDER_USE_METAL gates).
    if(APPLE AND COLLIDER_USE_METAL AND
       EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/v2/v2_metal_dispatch.mm")
        list(APPEND CORE_SOURCES src/gpu/v2/v2_metal_dispatch.mm)
        # Mark as Obj-C++ so CMake hands it to the right compiler.
        set_source_files_properties(src/gpu/v2/v2_metal_dispatch.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc -fobjc-weak")
    endif()
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
    # v1.4.1 UI: btc_balance.cpp uses libcurl when available. Define
    # COLLIDER_HAVE_CURL=1 so the header's #ifdef chooses the live
    # mempool.space code path; otherwise the stub returns nullopt.
    if(CURL_FOUND)
        target_link_libraries(collider_core CURL::libcurl)
        target_compile_definitions(collider_core PRIVATE COLLIDER_HAVE_CURL=1)
    endif()
    if(KANGAROO_METAL_GENERATED_INCLUDE_DIR)
        target_include_directories(collider_core PUBLIC
            ${KANGAROO_METAL_GENERATED_INCLUDE_DIR})
    endif()
    # Phase 11: link Metal + Foundation when v2_metal_dispatch.mm is in the build.
    if(APPLE AND COLLIDER_USE_METAL AND COLLIDER_PRO)
        target_link_libraries(collider_core
            ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
        )
    endif()
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
# JLP Protocol Codegen
# ============================================================================
# The C++ wire format header is generated from protocol/jlp.yaml by
# tools/codegen/jlp_codegen.py. The generated header lives at
# src/pool/jlp_wire_generated.hpp and is checked into the repo (so Free
# builds, which lack the codegen tool, can still compile). Pro builds
# regenerate it on every configure to catch IDL drift early.
set(JLP_IDL "${CMAKE_SOURCE_DIR}/protocol/jlp.yaml")
set(JLP_GEN_CPP "${CMAKE_SOURCE_DIR}/src/pool/jlp_wire_generated.hpp")
set(JLP_GEN_PY  "${CMAKE_SOURCE_DIR}/data/protocol/jlp_protocol_generated.py")
set(JLP_CODEGEN "${CMAKE_SOURCE_DIR}/tools/codegen/jlp_codegen.py")

if(EXISTS "${JLP_IDL}" AND EXISTS "${JLP_CODEGEN}")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        # Drift check at configure time: warn but don't fail (the
        # generated artifacts are committed; CI re-runs --check).
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${JLP_CODEGEN} --check
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE JLP_CHECK_RC
            OUTPUT_VARIABLE JLP_CHECK_OUT
            ERROR_VARIABLE  JLP_CHECK_ERR
        )
        if(NOT JLP_CHECK_RC EQUAL 0)
            message(WARNING
                "JLP codegen drift: regenerate via "
                "`python tools/codegen/jlp_codegen.py`. "
                "stderr: ${JLP_CHECK_ERR}")
        endif()

        # Per-build regen rule (only fires when the IDL or codegen change).
        add_custom_command(
            OUTPUT ${JLP_GEN_CPP} ${JLP_GEN_PY}
            COMMAND ${Python3_EXECUTABLE} ${JLP_CODEGEN}
            DEPENDS ${JLP_IDL} ${JLP_CODEGEN}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Regenerating JLP wire format from protocol/jlp.yaml"
            VERBATIM
        )
        add_custom_target(jlp_codegen DEPENDS ${JLP_GEN_CPP} ${JLP_GEN_PY})
    else()
        message(STATUS "Python3 not found; skipping JLP codegen (using committed generated files)")
    endif()
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
    if(TARGET jlp_codegen)
        add_dependencies(collider_pool jlp_codegen)
    endif()
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
        # v1.4.1 BW-DEDUP-3: bloom_filter.cu (impl A, double-hash on
        # raw bytes) was deleted in favour of h160_bloom_filter.cu
        # (impl B, MurmurHash3) which matches build_bloom's .blf
        # format. Bench picked B-texture as winner (3.29x on Ampere
        # 3060). See docs/v1.4.1-mod-inv-audit.md for the parallel
        # decision on mod_inv (kept multiple impls there because of
        # algorithmic divergence; bloom impls were genuinely
        # interchangeable so consolidation made sense).
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

        # Brain Wallet v2 unity TU (Phase 4..9, v1.4.0). All v2 kernels
        # are consolidated into src/gpu/v2/v2_kernels_unity.cu so nvlink
        # has a single device-symbol graph -- the alternative (one .cu
        # per kernel) overflows nvlink's default 1MB stack on Windows
        # given the cumulative collider_gpu symbol count.
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/v2/v2_kernels_unity.cu" AND
           NOT COLLIDER_DISABLE_V2)
            list(APPEND GPU_SOURCES src/gpu/v2/v2_kernels_unity.cu)
        endif()
    endif()

    # NOTE: src/gpu/v2/v2_orchestrator.cpp is compiled into collider_core,
    # not collider_gpu, so main.cpp can call it without pulling in CUDA on
    # Free builds. See the CORE_SOURCES list above.

    # Add RCKangaroo wrapper if enabled
    if(COLLIDER_USE_RCKANGAROO AND TARGET rckangaroo)
        list(APPEND GPU_SOURCES src/gpu/rckangaroo_wrapper.cu)
        # CUDA IKangarooBackend adapter -- only compiled with RCKangaroo
        # active. Lives in collider_gpu (rather than collider_core)
        # because it #include's rckangaroo_wrapper.hpp; that keeps the
        # core lib free of CUDA dependencies on Free builds.
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/cuda_rckangaroo_backend.cpp")
            list(APPEND GPU_SOURCES src/gpu/cuda_rckangaroo_backend.cpp)
        endif()
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

    # Phase 11 (v1.4.0): Brain Wallet v2 puzzle-only Metal shader.
    # 1:1 port of the CUDA puzzle-mode kernel; loaded at runtime via the
    # platform HAL. Only included on COLLIDER_PRO builds, mirroring the
    # CUDA-side gating.
    if(COLLIDER_PRO AND
       EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/v2/brain_wallet_v2.metal")
        list(APPEND METAL_SHADERS src/gpu/v2/brain_wallet_v2.metal)
    endif()

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

    # collider_core PUBLIC-links collider_platform (and the INTERFACE
    # fallback does the same), so collider_platform flows through
    # transitively when we list collider_core. Putting it ALSO in
    # MAIN_LIBS made Apple ld emit
    #    ld: warning: ignoring duplicate libraries: 'libcollider_platform.a'
    # because the link line carried -lcollider_platform twice. Drop the
    # explicit add; transitive resolution handles it.
    set(MAIN_LIBS Threads::Threads)

    if(TARGET collider_core)
        list(APPEND MAIN_LIBS collider_core)
    else()
        # Edge case: no CORE_SOURCES -> no collider_core target was created.
        # Pull in collider_platform directly so collider still links.
        list(APPEND MAIN_LIBS collider_platform)
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

    # KAT for range_bits_from_be in src/core/byte_codec.hpp. Pins the
    # canonical Bitcoin-puzzle ranges (75/135/160) to expected
    # range_bits values so a regression in the 256-bit subtraction +
    # bit_length helper would fail loudly. Used by Phase 1.4
    # CudaRCKangarooBackend::initialize() to size the kangaroo solver.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_range_bits.cpp")
        add_executable(test_range_bits tests/test_range_bits.cpp)
        # Header-only target -- byte_codec.hpp is inline-only.
        add_test(NAME RangeBitsKAT COMMAND test_range_bits)
    endif()

    # KAT for the rest of byte_codec.hpp: hex_encode_lower (~13 call
    # sites collapsed onto this in v1.4.0 phase 7.1) plus
    # be32_to_limbs_le / limbs_le_to_be32 roundtrip with the secp256k1
    # generator x-coordinate as the canonical test vector. Pure host
    # test, no GPU dependency.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_byte_codec.cpp")
        add_executable(test_byte_codec tests/test_byte_codec.cpp)
        add_test(NAME ByteCodecKAT COMMAND test_byte_codec)
    endif()

    # KAT for src/core/puzzle_config.hpp puzzle table: derive hash160
    # from each known solution_hex via cpu::compute_hash160 and verify
    # it matches the recorded target_h160_hex. Catches the data-
    # corruption class fixed in v1.4.0 phase 2.1 (mismatched addresses
    # + fabricated private keys). Pure host, runs everywhere.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_puzzle_table_kat.cpp")
        add_executable(test_puzzle_table_kat tests/test_puzzle_table_kat.cpp)
        add_test(NAME PuzzleTableKAT COMMAND test_puzzle_table_kat)
    endif()

    # Roundtrip KAT for KangarooWorkFile save/load (src/core/kangaroo.hpp).
    # The phase 1.8 mismatched-checkpoint rejection in
    # KangarooSolver::solve() depends on save/load being lossless;
    # this test pins down that contract.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_kangaroo_work_file.cpp")
        add_executable(test_kangaroo_work_file tests/test_kangaroo_work_file.cpp)
        add_test(NAME KangarooWorkFileKAT COMMAND test_kangaroo_work_file)
    endif()

    # Metal kangaroo throughput benchmark. Production-shaped config
    # (1024 kangaroos x 1024 steps x 32 rounds) so dispatch overhead
    # is amortized and the reported Mops/s reflects kernel throughput.
    # Used as the baseline for Phase 5 Montgomery batch inversion
    # before/after comparison. Mac-only.
    if(APPLE AND COLLIDER_USE_METAL AND
       EXISTS "${CMAKE_SOURCE_DIR}/tests/bench_metal_kangaroo.mm")
        add_executable(bench_metal_kangaroo tests/bench_metal_kangaroo.mm)
        set_target_properties(bench_metal_kangaroo PROPERTIES
            COMPILE_FLAGS "-fobjc-arc")
        target_link_libraries(bench_metal_kangaroo
            collider_core
            "-framework Foundation"
            "-framework Metal")
        # Not added as a ctest target (long-running, perf-only).
    endif()

    # Hashcat rule engine known-answer test. Pins the engine to externally
    # documented rule semantics so a regression vs. hashcat behaviour fails
    # loudly. Header-only, runs everywhere (no GPU / network required).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_rule_engine_kat.cpp" AND TARGET collider_core)
        add_executable(test_rule_engine_kat tests/test_rule_engine_kat.cpp)
        target_link_libraries(test_rule_engine_kat collider_core)
        add_test(NAME RuleEngineKAT COMMAND test_rule_engine_kat)
    endif()

    # Brain Wallet v2 host-side correctness test (Phase 9, v1.4.0).
    # The .cpp file has a CPU section that runs anywhere and a GPU section
    # gated on COLLIDER_USE_CUDA. The kernel itself is wired into
    # collider_gpu when COLLIDER_PRO is on. We always build the host
    # half so puzzle-target mask math is regression-tested.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_brain_wallet_v2.cpp" AND COLLIDER_PRO)
        add_executable(test_brain_wallet_v2 tests/v2/test_brain_wallet_v2.cpp)
        target_link_libraries(test_brain_wallet_v2 collider_core)
        if(TARGET collider_gpu)
            target_link_libraries(test_brain_wallet_v2 collider_gpu)
        endif()
        target_include_directories(test_brain_wallet_v2 PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME BrainWalletV2 COMMAND test_brain_wallet_v2)
        # Skip exit code 77 if no CUDA device available at runtime.
        set_tests_properties(BrainWalletV2 PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # Metal secp256k1 known-answer test (Mac only). Validates the field
    # arithmetic + point-add primitives in src/gpu/kangaroo.metal against
    # sipa/secp256k1 published priv-to-pub vectors. Returns 77 if no
    # Metal device is available so headless CI doesn't fail.
    if(APPLE AND COLLIDER_USE_METAL
       AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_metal_secp256k1.mm")
        add_executable(test_metal_secp256k1 tests/test_metal_secp256k1.mm)
        set_source_files_properties(tests/test_metal_secp256k1.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        target_link_libraries(test_metal_secp256k1
            ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
        target_include_directories(test_metal_secp256k1
            PRIVATE ${KANGAROO_METAL_GENERATED_INCLUDE_DIR})
        add_test(NAME MetalSecp256k1KAT COMMAND test_metal_secp256k1)
        set_tests_properties(MetalSecp256k1KAT PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # Metal Kangaroo smoke test: drives the kangaroo_step kernel through
    # the host dispatcher, asserts DP detection fires. Pairs with the
    # secp256k1 KAT to gate Mac Kangaroo shipping.
    if(APPLE AND COLLIDER_USE_METAL
       AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_metal_kangaroo_smoke.mm"
       AND EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/kangaroo_metal.mm")
        add_executable(test_metal_kangaroo_smoke
            tests/test_metal_kangaroo_smoke.mm)
        set_source_files_properties(tests/test_metal_kangaroo_smoke.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        # The Metal dispatcher already lives in collider_core (added to
        # CORE_SOURCES on Mac builds). Link the test against the lib so
        # we don't double-compile kangaroo_metal.mm and produce duplicate
        # symbols at link time.
        target_link_libraries(test_metal_kangaroo_smoke
            collider_core
            ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
        target_include_directories(test_metal_kangaroo_smoke
            PRIVATE ${CMAKE_SOURCE_DIR}/src)
        add_test(NAME MetalKangarooSmoke COMMAND test_metal_kangaroo_smoke)
        set_tests_properties(MetalKangarooSmoke PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # Metal fused puzzle pipeline KAT (v1.4.1). Runs privkey=1 / privkey=2
    # through the puzzle_search kernel and checks the recovered hash160
    # matches the published Bitcoin puzzle 1 / 2 vectors. A passing run
    # validates EC scalar mul, mod_inv, compress, SHA-256, RIPEMD-160,
    # and the compare-and-CAS logic end-to-end. Returns 77 if no Metal
    # device is present; ctest is configured with SKIP_RETURN_CODE=77.
    if(APPLE AND COLLIDER_USE_METAL
       AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_metal_puzzle_kat.mm"
       AND EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/puzzle_metal.mm")
        add_executable(test_metal_puzzle_kat
            tests/test_metal_puzzle_kat.mm)
        set_source_files_properties(tests/test_metal_puzzle_kat.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        target_link_libraries(test_metal_puzzle_kat
            collider_core
            ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
        target_include_directories(test_metal_puzzle_kat
            PRIVATE ${CMAKE_SOURCE_DIR}/src)
        add_test(NAME MetalPuzzleKAT COMMAND test_metal_puzzle_kat)
        set_tests_properties(MetalPuzzleKAT PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # Brain Wallet v2 host-side orchestrator test (Phase 9 follow-up).
    # Pure C++ test; we link collider_gpu too because v2_orchestrator.cpp
    # (compiled into collider_core) references kernels from collider_gpu
    # under #if COLLIDER_USE_CUDA -- the host code paths are dead in the
    # dry-run test, but the linker still needs the symbols.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_v2_orchestrator.cpp" AND COLLIDER_PRO)
        add_executable(test_v2_orchestrator tests/v2/test_v2_orchestrator.cpp)
        target_link_libraries(test_v2_orchestrator collider_core)
        if(TARGET collider_gpu)
            target_link_libraries(test_v2_orchestrator collider_gpu)
        endif()
        target_include_directories(test_v2_orchestrator PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME V2Orchestrator COMMAND test_v2_orchestrator)
    endif()

    # Phase 4: address-derivation CPU reference KATs.
    # Pure C++ -- header-only SHA-256 + RIPEMD-160 + hash160 reference,
    # plus the five address-type derivations. Validated against published
    # test vectors (BIP-49 / BIP-84 / BIP-86 / Bitcoin Core).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_address_derive_cpu.cpp" AND COLLIDER_PRO)
        add_executable(test_address_derive_cpu tests/v2/test_address_derive_cpu.cpp)
        target_include_directories(test_address_derive_cpu PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME AddressDeriveCPU COMMAND test_address_derive_cpu)
    endif()

    # Phase 5: weak-PRNG CPU reference KATs.
    # MT19937, Park-Miller LCG, MSVC rand(), java.util.Random + the
    # libbitcoin bx and Profanity vulnerable constructions. Validated
    # against the official MT reference table, Java spec, and the known
    # MSVC rand() trace.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_weak_prng_cpu.cpp" AND COLLIDER_PRO)
        add_executable(test_weak_prng_cpu tests/v2/test_weak_prng_cpu.cpp)
        target_include_directories(test_weak_prng_cpu PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME WeakPrngCPU COMMAND test_weak_prng_cpu)
    endif()

    # Phase 6: encoding-munger CPU reference KATs.
    # UTF-8 / UTF-16-LE / UTF-16-BE / UTF-32-LE / Latin-1 /
    # strip-non-ascii / upper-ascii / lower-ascii. Validated against
    # Python's stdlib codecs.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_encoding_munge_cpu.cpp" AND COLLIDER_PRO)
        add_executable(test_encoding_munge_cpu tests/v2/test_encoding_munge_cpu.cpp)
        target_include_directories(test_encoding_munge_cpu PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME EncodingMungeCPU COMMAND test_encoding_munge_cpu)
    endif()

    # Phase 7/8: SHA-512 + HMAC + PBKDF2 reference KATs.
    # Validated against NIST CAVS, RFC 4231 (HMAC), RFC 6070 (PBKDF2),
    # and the BIP-39 reference vector for "abandon abandon ... about".
    # These are the building blocks for the legacy KDF framework
    # (Phase 7) and Electrum / BIP-39 mnemonic kernels (Phase 8).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_pbkdf2_cpu.cpp" AND COLLIDER_PRO)
        add_executable(test_pbkdf2_cpu tests/v2/test_pbkdf2_cpu.cpp)
        target_include_directories(test_pbkdf2_cpu PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME Pbkdf2CPU COMMAND test_pbkdf2_cpu)
    endif()

    # Phase 8: Electrum v1 + v2 mnemonic CPU reference KATs.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_electrum_cpu.cpp" AND COLLIDER_PRO)
        add_executable(test_electrum_cpu tests/v2/test_electrum_cpu.cpp)
        target_include_directories(test_electrum_cpu PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME ElectrumCPU COMMAND test_electrum_cpu)
    endif()

    # Phase 10: license cache lifecycle tests.
    # Validates the HMAC-bound cache that gates Pro features after a
    # successful remote validation. No filesystem / network -- the test
    # surface (make_cache_payload / compute_cache_hmac / verify_cache_entry)
    # is the algorithm under test.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_license_cache.cpp"
       AND TARGET collider_license)
        add_executable(test_license_cache tests/test_license_cache.cpp)
        target_link_libraries(test_license_cache collider_license)
        add_test(NAME LicenseCache COMMAND test_license_cache)
        set_tests_properties(LicenseCache PROPERTIES SKIP_RETURN_CODE 77)
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

    # JLP wire generated-vs-legacy drift detector. Static_asserts only,
    # plus three runtime byte-layout sanity checks. No threads, no sockets.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_wire_generated.cpp" AND TARGET collider_pool)
        add_executable(test_jlp_wire_generated tests/test_jlp_wire_generated.cpp)
        target_link_libraries(test_jlp_wire_generated collider_pool)
        if(TARGET jlp_codegen)
            add_dependencies(test_jlp_wire_generated jlp_codegen)
        endif()
        add_test(NAME JLPWireGenerated COMMAND test_jlp_wire_generated)
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

        # Device HMAC-SHA512 KAT (RFC 4231 TC1/TC2/TC4/TC6). Pins the
        # output of device::hmac_sha512 in src/gpu/v2/device_hashes.cuh
        # against the published vectors -- catches regressions in the
        # streaming-rewrite stack-reduction refactor.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_device_hmac_sha512.cu")
            add_executable(test_device_hmac_sha512 tests/test_device_hmac_sha512.cu)
            set_target_properties(test_device_hmac_sha512 PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_include_directories(test_device_hmac_sha512
                PRIVATE ${CMAKE_SOURCE_DIR}/src)
            target_link_libraries(test_device_hmac_sha512 CUDA::cudart)
            add_test(NAME DeviceHmacSha512KAT COMMAND test_device_hmac_sha512)
            set_tests_properties(DeviceHmacSha512KAT PROPERTIES SKIP_RETURN_CODE 77)
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

    # v1.4.1 BW-DEDUP-3: bench_bloom_ab.cu was used to pick between
    # the two bloom impls. After the verdict (B-texture wins), the
    # losing impl was deleted and the bench retired -- its purpose
    # is served. A future B-global-vs-B-texture bench would be a
    # different harness (h160_bloom_filter.cu has no CUDA insert
    # wrapper, so the throughput-only comparison would look quite
    # different).
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
