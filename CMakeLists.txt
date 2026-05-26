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
    # std::jthread / std::stop_token are availability-gated in Apple's libc++.
    # macOS 15.0 (Sequoia) is the first release where the gate opens; 14.0
    # (Sonoma) still lacks the _LIBCPP_AVAILABILITY_HAS_JTHREAD attribute.
    # Setting this before the project() declaration ensures the Xcode toolchain
    # picks it up for all targets, including the SDK sysroot selection.
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version" FORCE)
    endif()
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

# Opt-in diagnostic: device-side printf bounds-check at every bloom probe
# in the fused brain-wallet kernel. Used to narrow down an intermittent
# cudaErrorIllegalAddress on the V2 multi_addr path that does not
# reproduce in the existing ctest harness. The check is OFF by default
# because printf in a hot kernel costs ~20-30% throughput and the test
# is only useful while diagnosing. Enable per-build with
# `-DCOLLIDER_DEBUG_FUSED_BOUNDS=ON`. The macro is consumed by
# src/gpu/fused_pipeline.cu::bloom_check_inline.
option(COLLIDER_DEBUG_FUSED_BOUNDS
       "Enable device-side bloom-probe bounds check in fused kernel (diagnostic)"
       OFF)
if(COLLIDER_DEBUG_FUSED_BOUNDS)
    add_compile_definitions(COLLIDER_DEBUG_FUSED_BOUNDS=1)
    message(STATUS "Fused kernel bounds-check instrumentation ENABLED (diagnostic build)")
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
        # RTX 2060/2070/2080 (Turing): SM 7.5
        # RTX 3060/3070/3080/3090 (Ampere): SM 8.6
        # RTX 4090 (Ada Lovelace): SM 8.9
        # RTX 5090 / RTX PRO 6000 (Blackwell Desktop): SM 12.0
        # NOTE: SM 10.0 is datacenter Blackwell (B100/B200), not desktop.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120")
            message(STATUS "Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()

        # T4.2: surface the compiled-in CUDA architecture list to runtime so
        # gpu_detection.cpp can warn when the active device's SM is missing
        # from the binary. The list flows through as a comma-separated string
        # of "X.Y" tokens (e.g. "7.5,8.6,8.9,12.0") so the C++ side can
        # tokenize without parsing the raw "75;86;89;120" format. Building
        # the formatted string here keeps the parsing burden off the runtime.
        set(_collider_cuda_arch_csv "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(LENGTH "${_arch}" _arch_len)
            if(_arch_len GREATER_EQUAL 2)
                math(EXPR _minor_pos "${_arch_len} - 1")
                string(SUBSTRING "${_arch}" 0 ${_minor_pos} _major)
                string(SUBSTRING "${_arch}" ${_minor_pos} 1 _minor)
                if(_collider_cuda_arch_csv STREQUAL "")
                    set(_collider_cuda_arch_csv "${_major}.${_minor}")
                else()
                    set(_collider_cuda_arch_csv "${_collider_cuda_arch_csv},${_major}.${_minor}")
                endif()
            endif()
        endforeach()
        add_compile_definitions(COLLIDER_CUDA_ARCH_LIST="${_collider_cuda_arch_csv}")

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
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2 /GL /EHsc /Zi")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /DEBUG")
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

# ============================================================================
# FTXUI - C++17 functional terminal UI library for the brainwallet TUI
# ============================================================================
# Wired by Wave 1 builder-foundation. Pinned to v5.0.0 (the first tagged
# release with the FTXUI_BUILD_* options that Wave 1 needs to disable
# examples / docs / tests). Header-light enough to live under FetchContent
# (no separate vcpkg entry). GIT_SHALLOW shaves the clone time on
# configure since we never need history beyond the pinned tag.
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG v5.0.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ftxui)

# FTXUI's cmake/ftxui_set_options.cmake attaches `/utf-8` to every
# library as PUBLIC under MSVC. PUBLIC means the flag propagates to
# every consumer, including the `collider` executable, which also
# compiles cuda_stub.cu via nvcc. nvcc on Windows interprets `/utf-8`
# as an input-file path (leading `/`), trips its "single input file
# for non-link phase" guard, and aborts. The fix is to rewrap the
# PUBLIC compile options of each FTXUI target so `/utf-8` only fires
# on CXX translation units, not CUDA.
foreach(_ftxui_target screen dom component)
    if(TARGET ${_ftxui_target} AND MSVC)
        get_target_property(_opts ${_ftxui_target} INTERFACE_COMPILE_OPTIONS)
        if(_opts)
            set(_new_opts "")
            foreach(_opt ${_opts})
                if(_opt STREQUAL "/utf-8")
                    list(APPEND _new_opts
                        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
                else()
                    list(APPEND _new_opts ${_opt})
                endif()
            endforeach()
            set_target_properties(${_ftxui_target} PROPERTIES
                INTERFACE_COMPILE_OPTIONS "${_new_opts}")
        endif()
    endif()
endforeach()

# Interface library that the collider binary will link against. Bundles
# the three FTXUI sub-targets (screen / dom / component) so downstream
# consumers list one name instead of three.
add_library(collider_tui INTERFACE)
target_link_libraries(collider_tui INTERFACE
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# ============================================================================
# NVML - NVIDIA Management Library for GPU telemetry
# ============================================================================
# NVML is loaded at runtime via dlopen / GetProcAddress in
# src/platform/nvml_query.cpp, so we don't strictly need to link the
# library here. What we DO need is the include directory (for nvml.h)
# and the COLLIDER_HAS_NVML compile definition so #ifdef gating in the
# wrapper compiles. Mac builds skip NVML entirely.
set(NVML_LIB NOTFOUND)
set(NVML_INCLUDE_DIR NOTFOUND)
if(NOT APPLE)
    if(WIN32 AND CUDAToolkit_FOUND)
        # On Windows the NVIDIA driver ships nvml.dll in System32 and the
        # CUDA Toolkit ships nvml.lib + nvml.h. Look in CUDAToolkit's
        # library / include dirs first, fall back to CUDA_PATH env var.
        find_library(NVML_LIB nvml
            HINTS
                "${CUDAToolkit_LIBRARY_DIR}"
                "$ENV{CUDA_PATH}/lib/x64"
        )
        find_path(NVML_INCLUDE_DIR nvml.h
            HINTS
                "${CUDAToolkit_INCLUDE_DIRS}"
                "$ENV{CUDA_PATH}/include"
        )
    elseif(UNIX AND CUDAToolkit_FOUND)
        # Linux: libnvidia-ml is shipped by the NVIDIA driver, not the
        # toolkit. The toolkit ships nvml.h.
        find_library(NVML_LIB nvidia-ml)
        find_path(NVML_INCLUDE_DIR nvml.h
            HINTS "${CUDAToolkit_INCLUDE_DIRS}"
        )
    endif()
endif()

if(NVML_INCLUDE_DIR)
    message(STATUS "NVML support enabled (include: ${NVML_INCLUDE_DIR})")
    if(NVML_LIB)
        message(STATUS "  NVML library: ${NVML_LIB}")
    else()
        message(STATUS "  NVML library: not found (loaded at runtime via dlopen / LoadLibrary)")
    endif()
else()
    message(STATUS "NVML support disabled (nvml.h not found)")
endif()

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
    # Q7: gpu-layer instrumentation callback storage. Plain C++ (no CUDA),
    # holds the two atomic function-pointer slots that fused_pipeline.cu
    # and the runtime perf adapter share. Was previously only built into
    # collider_gpu under the CUDA branch, which left register_callbacks
    # undefined at link time on Metal (perf_instrumentation.cpp's static
    # initializer calls it from collider_core). Lives here so every
    # backend gets the symbol regardless of the GPU code path.
    src/gpu/instrumentation_hook.cpp
    # v1.4.2 quality: per-session log + atomic session_state.json + OS
    # crash handler. Extends core/logger.hpp's rotating singleton with
    # a one-file-per-invocation log, a structured JSON snapshot a
    # recovery tool can read, and a SetUnhandledExceptionFilter /
    # sigaction handler that dumps the snapshot on SEGV/SEH.
    src/core/session_log.cpp
    # CLI argument parsing extracted from main.cpp so the
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
    # v1.4.2 quality: internal decomposition of the puzzle-mode runtime.
    # Helpers extracted from the inline body of run_puzzle_mode (worklist
    # build, GPU calibration trigger, per-puzzle target resolution,
    # algorithm selection, search-space banner). Header is internal to
    # the runtime layer; puzzle_solver.hpp's public surface is unchanged.
    src/runtime/puzzle_solver_helpers.cpp
    # v1.4.2 quality: kangaroo solve-path (RCKangaroo -> MultiGPU -> CPU
    # fallback) extracted from run_puzzle_mode's inline KANGAROO MODE
    # block. Preserves --resume-kangaroo, SaveKangsHost/InitKangsHost
    # wiring, and secure_wipe on every recovered private key.
    src/runtime/puzzle_solver_kangaroo.cpp
    # v1.5.x Phase F2: GPU BSGS standalone-challenge path. Routed from
    # the kangaroo dispatcher when args.solver == "bsgs".
    src/runtime/puzzle_solver_bsgs.cpp
    # v1.4.2 quality: brute-force solve-path (>128-bit reject, GPU brute,
    # CPU brute, final stats) extracted from run_puzzle_mode. Preserves
    # SearchState v4 position_full[0..3] layout and SIGINT save semantics.
    src/runtime/puzzle_solver_bruteforce.cpp
    # v1.4.2 quality: Free SHA-256-only benchmark path extracted from
    # run_benchmark. The Pro --benchmark path stays inline as a thin
    # delegation to runtime/bench_pipeline (the source of truth for
    # the PRO docs' throughput numbers).
    src/runtime/puzzle_solver_benchmark.cpp
    # mempool.space balance check spawned after a true hit. Shared by
    # puzzle_solver.cpp and brain_wallet_runner.cpp via runtime/balance.hpp.
    src/runtime/balance.cpp
    # GPU detection helper extracted from main.cpp so the dispatcher
    # (main.cpp) and the runtime drivers can both link against
    # detect_gpus() without main.cpp owning the body.
    src/runtime/gpu_detection.cpp
    # v1.4.1 UI: live BTC balance lookup against mempool.space for the
    # solved-puzzle banner. libcurl-gated; if CURL_FOUND was false at
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
    # Metal IKangarooBackend adapter (gated to the same conditions as
    # the underlying KangarooMetalSolver above).
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
    # --brute incremental bruteforce generator. PRO-only because the
    # brain-wallet runtime that consumes it is PRO-only.
    list(APPEND CORE_SOURCES src/generators/bruteforce_generator.cpp)
    # v1.4.2 C.2: brain_wallet_engine.{hpp,cpp} deleted; it had zero callers
    # in the codebase and its algorithmic content (PCFG, Markov, keyboard
    # walks, combinator) was already implemented in StreamingBrainWallet.
    # QWERTZ + AZERTY keyboard-walk seeds were merged into
    # init_keyboard_walk_mode(); crypto-vocab tables were moved to
    # data/wordlists/brain_wallet_terms.txt for operator use as a
    # --wordlist alongside the rule engine.
    # v1.4.1 A.3: brain wallet runtime driver extracted from main.cpp.
    # Whole TU is COLLIDER_PRO-gated; the file itself wraps everything
    # in #ifdef COLLIDER_PRO defense-in-depth.
    list(APPEND CORE_SOURCES src/runtime/brain_wallet_runner.cpp)
    # v1.4.2 quality: .blf bloom-filter loader extracted from
    # brain_wallet_runner.cpp. Independent TU so the file-IO + header
    # validation can be reviewed in isolation and reused by both the
    # primary brain-wallet bloom load and the --bloom-tight side load
    # without duplicating the validation sequence.
    list(APPEND CORE_SOURCES src/runtime/bloom_loader.cpp)
    # v1.4.2 quality: BrainWalletSearchState payload builder extracted from
    # brain_wallet_runner.cpp. SaveStateInputs + build_brain_wallet_search_state
    # are the single source-of-truth assembly routine consumed by all four
    # save-state call sites (explicit / paused / periodic / final).
    list(APPEND CORE_SOURCES src/runtime/brain_wallet_setup.cpp)
    # T3.14 (2026-05-17): WarpWallet CPU scan path extracted from
    # brain_wallet_runner.cpp. Independent translation unit so the CPU
    # scrypt-based scan loop can be reviewed without GPU-pipeline
    # context. Pro-only (same #ifdef guard as the rest of brainwallet).
    list(APPEND CORE_SOURCES src/runtime/warpwallet_runner.cpp)
    # v1.5.x: BIP brainwallet scanner. Iterates BIP-39 candidate phrases,
    # derives BIP-32 children across every historical + modern path
    # (pre-BIP-44 / Electrum / MultiBit / BIP-44/49/84), probes each
    # address against the loaded bloom filter. Pro-only; selected via
    # --bip-scan on the CLI.
    list(APPEND CORE_SOURCES src/runtime/bip_scanner_runner.cpp)
    # GPU dispatcher for BIP scan: routes derived priv keys through
    # the existing MultiAddressSession kernel (secp256k1 + hash160 +
    # bloom probe) so the per-pubkey work runs on GPU instead of CPU.
    # Multi-device aware (one worker thread per GPU in args.gpu_ids).
    list(APPEND CORE_SOURCES src/runtime/bip_gpu_dispatcher.cpp)
    # Full-pipeline benchmark core. Shared between the --benchmark CLI
    # path (puzzle_solver.cpp) and the standalone benchmarks/
    # bench_gpu_pipeline.cu binary. Pro-only because it depends on the
    # fused brain-wallet kernels; the Free benchmark stays SHA-256-only.
    list(APPEND CORE_SOURCES src/runtime/bench_pipeline.cpp)
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

    # Runtime support (runtime.yml writeback, perf instrumentation,
    # empty-hit writer) and plugin runner / registry land in
    # collider_core because brain_wallet_runner.cpp consumes them.
    # scan_state and runtime_control are header-only (atomics + an inline
    # singleton accessor); they have no compiled translation units.
    list(APPEND CORE_SOURCES
        src/runtime/runtime_config_yaml.cpp
        src/runtime/perf_instrumentation.cpp
        # Tier 1 perf F-log (builder-perf): background writer for
        # --track-empty-hits, replaces the inline log_empty_hit that
        # opened+closed found-empty.txt on the scan thread per hit.
        src/runtime/empty_hit_writer.cpp
        # Phase 5 (builder-plugins: plugin-protocol-registry)
        src/plugins/plugin_protocol.cpp
        src/plugins/plugin_registry.cpp
        # Phase 5 (builder-threading: plugin-runner)
        src/plugins/plugin_runner.cpp
    )

    # FTXUI panels + widgets. These translation units #include FTXUI
    # headers, so collider_core has to PUBLIC-link collider_tui (handled
    # in the dependency block below).
    list(APPEND CORE_SOURCES
        # Phase 1 (builder-foundation: tui-app-skeleton)
        src/ui/tui/tui_app.cpp
        src/ui/tui/tui_launcher.cpp
        src/ui/tui/theme.cpp
        # Carbon-fiber boot banner (2026-05-16). Standalone animation
        # printed before TuiApp::start(); not part of the FTXUI panel
        # composition but lives alongside the TUI sources because it
        # is the TUI mode's opening visual.
        src/ui/tui/boot_banner.cpp
        # Phase 1 (builder-threading: cooked-mode-guard)
        src/ui/tui/cooked_mode_guard.cpp
        # Phase 4 (builder-threading: runtime-control-input)
        src/ui/tui/keybindings.cpp
        src/ui/tui/input_handler.cpp
        # Phase 1..6 panels (builder-panels)
        src/ui/tui/panels/header_panel.cpp
        src/ui/tui/panels/status_panel.cpp
        src/ui/tui/panels/footer_panel.cpp
        src/ui/tui/panels/gpu_panel.cpp
        src/ui/tui/panels/performance_panel.cpp
        src/ui/tui/panels/plugins_panel.cpp
        src/ui/tui/panels/recent_hits_panel.cpp
        src/ui/tui/panels/help_overlay.cpp
        src/ui/tui/panels/settings_panel.cpp
        src/ui/tui/panels/wordlist_composer.cpp
        src/ui/tui/panels/wordlist_picker.cpp
        src/ui/tui/panels/bloom_picker.cpp
        # TR-1: TUI-native main menu (replaces cout banner+numbered list).
        src/ui/tui/menu/main_menu.cpp
        # TR-1-full: TUI mode-config modals (BIP confirm + puzzle picker).
        src/ui/tui/menu/mode_config.cpp
        # Phase 1..3 widgets (builder-panels)
        src/ui/tui/widgets/sparkline.cpp
        src/ui/tui/widgets/progress_bar.cpp
        src/ui/tui/widgets/histogram.cpp
        # Tier 2 D2 hero metrics (builder-panels, 2026-05-15)
        src/ui/tui/widgets/big_digits.cpp
    )

    # Phase 2 (builder-gpu): NVML wrapper + GPU telemetry abstraction.
    # nvml_query.cpp + gpu_telemetry_cuda.cpp on Windows + Linux; the
    # .mm Metal variant on Mac.
    if(NOT APPLE)
        list(APPEND CORE_SOURCES
            src/platform/nvml_query.cpp
            src/platform/gpu_telemetry_cuda.cpp
        )
    else()
        list(APPEND CORE_SOURCES
            src/platform/gpu_telemetry_metal.mm
        )
        set_source_files_properties(src/platform/gpu_telemetry_metal.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
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
    # v1.5.x: BIP scanner runtime + bip32.hpp need OpenSSL for PBKDF2 +
    # HMAC-SHA512. OpenSSL is already linked for collider_pool (TLS);
    # propagate the link + define here so collider_core's BIP code paths
    # can call PKCS5_PBKDF2_HMAC and HMAC() directly.
    if(OPENSSL_FOUND)
        target_link_libraries(collider_core OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(collider_core PUBLIC COLLIDER_HAS_OPENSSL)
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
    # NVML: feed the include dir + define to nvml_query.cpp. The
    # library is loaded at runtime (dlopen / LoadLibraryA), so direct
    # linking is unnecessary, but the header has to be on the include
    # path or the wrapper can't see nvmlReturn_t etc.
    if(NVML_INCLUDE_DIR AND COLLIDER_PRO)
        target_include_directories(collider_core PRIVATE ${NVML_INCLUDE_DIR})
        target_compile_definitions(collider_core PUBLIC COLLIDER_HAS_NVML=1)
    endif()
    # TUI: PUBLIC link FTXUI through collider_tui so the panel /
    # widget translation units in collider_core can include FTXUI
    # headers, and downstream consumers (the collider executable) get
    # the same headers transitively without listing collider_tui again.
    if(COLLIDER_PRO AND TARGET collider_tui)
        target_link_libraries(collider_core collider_tui)
    endif()
    # session_log.cpp's Windows crash filter uses StackWalk64 + SymFromAddr
    # from dbghelp.dll. The library is part of the Windows SDK so no
    # find_package is needed; the link name is "dbghelp" on every MSVC
    # toolchain. Non-Windows builds skip this branch and fall through to
    # the POSIX backtrace() path in session_log.cpp which has no extra
    # link requirements (glibc backtrace is in libc itself).
    if(WIN32)
        target_link_libraries(collider_core dbghelp)
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
# COLLIDER_PRO=ON is a hard contract: license_gate.cpp is unconditionally
# compiled into the Pro build (see CORE_SOURCES append at line ~700) and
# unconditionally calls collider::license::validate / cache_path. If
# CURL_FOUND or OPENSSL_FOUND is false the link silently failed with
# undefined-reference errors -- the failure mode we hit on v1.4.4's first
# Linux release build, where libcurl4-openssl-dev was not in the apt
# install list. Fail at configure-time with an actionable error so CI and
# operators see the missing dep immediately instead of after a 30-minute
# compile.
if(COLLIDER_PRO AND EXISTS "${CMAKE_SOURCE_DIR}/src/license/license_check.cpp")
    if(NOT CURL_FOUND OR NOT OPENSSL_FOUND)
        message(FATAL_ERROR
            "COLLIDER_PRO=ON requires libcurl and OpenSSL "
            "(CURL_FOUND=${CURL_FOUND}, OPENSSL_FOUND=${OPENSSL_FOUND}). "
            "Ubuntu/Debian: apt-get install libcurl4-openssl-dev libssl-dev. "
            "macOS: brew install curl openssl. "
            "Windows: vcpkg.json declares both deps; ensure CMake "
            "bootstraps the toolchain (the in-tree CMakeLists auto-uses "
            "VCPKG_ROOT or downloads vcpkg on demand).")
    endif()
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
        # T0.3.a (A-tier wave 1): src/gpu/pipeline.cu removed. It hosted
        # the GPUPipeline + AsyncPipelineRunner orchestrator that was
        # never instantiated in production (the brain-wallet runner uses
        # brain_wallet_gpu.cpp + fused_pipeline.cu directly; the v2
        # multi-address bridge was dead in v1.4.x; the bench driver
        # calls the individual stage wrappers, not the orchestrator).
        # The file was also a passthrough caller of the deleted
        # secp256k1_batch_mul (windowed-mul double-shift bug); removing
        # it was the only way to make the symbol-deletion clean.
        src/gpu/fused_pipeline.cu
        src/gpu/puzzle_gpu.cu
        src/gpu/puzzle_optimized.cu
        src/gpu/bsgs_solver_gpu.cu
        src/gpu/kangaroo_kernel.cu
        # NOTE: src/gpu/instrumentation_hook.cpp lives in CORE_SOURCES so
        # the Metal backend also picks it up; without that, Mac Pro
        # linking errors with "undefined symbol: register_callbacks"
        # called from perf_instrumentation.cpp's static initializer.
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
            # Repair R5+R8: mega_fused_kernel.cu removed. The production
            # brainwallet path runs through gpu_rules.cu's
            # gpu_apply_rules_cross_product (rule expansion) and
            # fused_pipeline.cu's brain_wallet_fused_kernel_fixed_stride
            # (crypto+bloom); the mega-fused kernel was a never-invoked
            # alternative.
        )

        # Brain Wallet v2 unity TU (Phase 4..9, v1.4.0). All v2 kernels
        # are consolidated into src/gpu/v2/v2_kernels_unity.cu so nvlink
        # has a single device-symbol graph; the alternative (one .cu
        # per kernel) overflows nvlink's default 1MB stack on Windows
        # given the cumulative collider_gpu symbol count.
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/v2/v2_kernels_unity.cu" AND
           NOT COLLIDER_DISABLE_V2)
            list(APPEND GPU_SOURCES src/gpu/v2/v2_kernels_unity.cu)
        endif()

        # BIP-39 PBKDF2-HMAC-SHA512 kernel. Replaces OpenSSL's CPU
        # PBKDF2 path on the hot BIP scan loop (~1ms per phrase on
        # CPU; GPU dispatch lets one device do 2-5k phrases/sec on
        # consumer Ampere/Ada, multiplied per device in args.gpu_ids).
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/bip39_pbkdf2.cu")
            list(APPEND GPU_SOURCES src/gpu/bip39_pbkdf2.cu)
        endif()
    endif()

    # NOTE: src/gpu/v2/v2_orchestrator.cpp is compiled into collider_core,
    # not collider_gpu, so main.cpp can call it without pulling in CUDA on
    # Free builds. See the CORE_SOURCES list above.

    # Add RCKangaroo wrapper if enabled
    if(COLLIDER_USE_RCKANGAROO AND TARGET rckangaroo)
        list(APPEND GPU_SOURCES src/gpu/rckangaroo_wrapper.cu)
        # CUDA IKangarooBackend adapter; only compiled with RCKangaroo
        # active. Lives in collider_gpu (rather than collider_core)
        # because it #include's rckangaroo_wrapper.hpp; that keeps the
        # core lib free of CUDA dependencies on Free builds.
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/cuda_rckangaroo_backend.cpp")
            list(APPEND GPU_SOURCES src/gpu/cuda_rckangaroo_backend.cpp)
        endif()
        # v1.4.2 builder-final: standalone-side RCKangaroo IKangarooBackend
        # adapter. Lives alongside the pool-side adapter so the standalone
        # puzzle runner gets save/load via the patched third_party hooks.
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/rc_kangaroo_backend.cpp")
            list(APPEND GPU_SOURCES src/gpu/rc_kangaroo_backend.cpp)
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

    # B1 wire-v4: keygen tool. Generates a fresh secp256k1 worker
    # identity, writes 0600-permissioned WIF, prints the bech32 address.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/generate_worker_key.cpp"
       AND TARGET collider_core AND OPENSSL_FOUND)
        add_executable(generate_worker_key
            src/tools/generate_worker_key.cpp
        )
        target_link_libraries(generate_worker_key
            collider_core
            OpenSSL::SSL OpenSSL::Crypto
            Threads::Threads
        )
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

    # v1.4.2: Bitcoin Core full-node address extractor for the
    # --track-empty-hits workflow. Parses blk*.dat files directly; no
    # bitcoind needed. Self-contained: just <filesystem> + the C++ stdlib.
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/extract_node_addresses.cpp")
        add_executable(extract_node_addresses
            src/tools/extract_node_addresses.cpp
        )
        target_link_libraries(extract_node_addresses
            Threads::Threads
        )
        # std::filesystem needs explicit linkage on older libstdc++.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
           CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
            target_link_libraries(extract_node_addresses stdc++fs)
        endif()
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

    # brainwallet TUI: collider_tui is an INTERFACE library
    # bundling ftxui::screen / dom / component. PRO-only since the TUI
    # is wired into the brain wallet runtime.
    if(COLLIDER_PRO AND TARGET collider_tui)
        list(APPEND MAIN_LIBS collider_tui)
    endif()

    if(COLLIDER_BACKEND STREQUAL "CUDA")
        list(APPEND MAIN_LIBS CUDA::cudart)
    elseif(COLLIDER_BACKEND STREQUAL "METAL")
        list(APPEND MAIN_LIBS ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
    endif()

    target_link_libraries(collider ${MAIN_LIBS})

    # TR-1-full (2026-05-23): when building the Pro edition, the output
    # exe should be named collider_pro.exe so the build_pro/ directory
    # contract matches the build script + Open Brain references. The
    # CMake target stays named "collider" (changing it would ripple
    # through every set_target_properties / target_link_libraries call
    # above and below); we only override the linked output filename.
    if(COLLIDER_PRO)
        set_target_properties(collider PROPERTIES
            OUTPUT_NAME "collider_pro"
        )
    endif()

    # Stage data/ next to the built exe (post-build copy). The BIP
    # scanner's data_search_roots() walks the exe directory first, so
    # dropping data/bip39/english.txt next to collider_pro.exe makes
    # the scanner work out of the box on a fresh install. The CMake
    # source-to-build copy is a single recursive step that runs on
    # every build; copy_directory is idempotent so the cost is one
    # mtime check per file.
    if(EXISTS "${CMAKE_SOURCE_DIR}/data")
        add_custom_command(TARGET collider POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/data"
                "$<TARGET_FILE_DIR:collider>/data"
            COMMENT "Staging data/ (BIP-39 dictionaries + assets) next to exe"
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/wordlists")
        add_custom_command(TARGET collider POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/wordlists"
                "$<TARGET_FILE_DIR:collider>/wordlists"
            COMMENT "Staging wordlists/ (candidate phrase examples) next to exe"
        )
    endif()

    # For CUDA separable compilation, the final executable must resolve device symbols
    if(COLLIDER_BACKEND STREQUAL "CUDA")
        set_target_properties(collider PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_RUNTIME_LIBRARY Shared
        )
        # build fix: CMake + Ninja + nvcc on Windows generates a
        # compile-PDB flag of the form `-Xcompiler=-FdCMakeFiles\collider.dir\,-FS`.
        # The trailing backslash before the comma confuses nvcc's `-Xcompiler`
        # comma-list parser ("A single input file is required for a non-link
        # phase when an outputfile is specified"). Setting an explicit
        # COMPILE_PDB_NAME gives the flag a real filename component so the
        # trailing-backslash-then-comma pattern never appears.
        if(MSVC)
            set_target_properties(collider PROPERTIES
                COMPILE_PDB_NAME "collider"
                COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
            )
        endif()
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

    # v1.4.2: per-GPU work balancer used by the brain wallet runtime to
    # weight each GPU's slice by its rolling EMA throughput. Header-only
    # (no link dependencies); exercised with synthetic samples so the
    # test stays fast and runs on every backend.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_work_balancer.cpp")
        add_executable(test_work_balancer tests/test_work_balancer.cpp)
        add_test(NAME WorkBalancer COMMAND test_work_balancer)
    endif()

    # v1.4.2 A.1: regression for Wild1-Wild2 kangaroo collision recovery.
    # Header-only against crypto_cpu.hpp; no link dependencies.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_mod_half.cpp")
        add_executable(test_mod_half tests/test_mod_half.cpp)
        add_test(NAME ModHalf COMMAND test_mod_half)
    endif()

    # T3.2 (2026-05-17): WarpWallet known-answer + property test
    # (header-only crypto). Renamed back to test_warpwallet_kat.cpp now
    # that real Keybase reference vectors are pinned alongside the
    # property checks; see the test file header for the
    # rename-and-restore rationale.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_warpwallet_kat.cpp")
        add_executable(test_warpwallet_kat tests/test_warpwallet_kat.cpp)
        add_test(NAME WarpWalletKAT COMMAND test_warpwallet_kat)
        # scrypt(N=2^18) is intentionally slow; allow a generous timeout.
        # 3 KAT derivations + 3 property derivations = 6-18s in practice.
        set_tests_properties(WarpWalletKAT PROPERTIES TIMEOUT 60)
    endif()

    # v1.4.2 L.2: BIP-39 mnemonic validation KAT.
    # Run from the source tree so the test can locate the
    # data/crypto/bip39_english.txt wordlist.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip39_validate.cpp")
        add_executable(test_bip39_validate tests/test_bip39_validate.cpp)
        add_test(NAME Bip39Validate COMMAND test_bip39_validate
                 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(Bip39Validate PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # T1-E: BIP-39 trezor PBKDF2-HMAC-SHA512 seed vectors.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip39_pbkdf2_kat.cpp"
       AND TARGET collider_core)
        add_executable(test_bip39_pbkdf2_kat tests/test_bip39_pbkdf2_kat.cpp)
        target_link_libraries(test_bip39_pbkdf2_kat collider_core)
        add_test(NAME Bip39Pbkdf2Kat COMMAND test_bip39_pbkdf2_kat)
    endif()

    # BIP-39 PBKDF2 GPU kernel KAT: cross-checks GPU output byte-for-
    # byte against the canonical CPU OpenSSL output. Skipped (exit 77)
    # when CUDA is not available.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip39_pbkdf2_gpu_kat.cpp"
       AND TARGET collider_core AND COLLIDER_BACKEND STREQUAL "CUDA")
        add_executable(test_bip39_pbkdf2_gpu_kat
            tests/test_bip39_pbkdf2_gpu_kat.cpp)
        target_link_libraries(test_bip39_pbkdf2_gpu_kat
            collider_core collider_gpu)
        add_test(NAME Bip39Pbkdf2GpuKat
                 COMMAND test_bip39_pbkdf2_gpu_kat)
        set_tests_properties(Bip39Pbkdf2GpuKat
                             PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # BipGpuDispatcher lifecycle + hit-routing test. Verifies init
    # failure paths (nullptr on_hit, empty gpu_ids), end-to-end hit
    # callback wiring (seeded bloom + matching priv => exactly one
    # on_hit fires), shutdown idempotency, and destructor-without-
    # shutdown cleanup. Skipped (exit 77) when CUDA is not available.
    # PRO-only since the dispatcher class itself is gated on
    # COLLIDER_PRO.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip_gpu_dispatcher.cpp"
       AND TARGET collider_core AND COLLIDER_BACKEND STREQUAL "CUDA"
       AND COLLIDER_PRO)
        add_executable(test_bip_gpu_dispatcher
            tests/test_bip_gpu_dispatcher.cpp)
        target_link_libraries(test_bip_gpu_dispatcher
            collider_core collider_gpu)
        add_test(NAME BipGpuDispatcher
                 COMMAND test_bip_gpu_dispatcher)
        set_tests_properties(BipGpuDispatcher
                             PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # T1-F: BIP-49 P2SH-P2WPKH hash160 spec vector.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip49_p2sh_p2wpkh_kat.cpp"
       AND TARGET collider_core)
        add_executable(test_bip49_p2sh_p2wpkh_kat
            tests/test_bip49_p2sh_p2wpkh_kat.cpp)
        target_link_libraries(test_bip49_p2sh_p2wpkh_kat collider_core)
        add_test(NAME Bip49P2shP2wpkhKat COMMAND test_bip49_p2sh_p2wpkh_kat)
    endif()

    # T1-D: BIP scanner end-to-end smoke test.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip_scan_runner_smoke.cpp"
       AND TARGET collider_core)
        add_executable(test_bip_scan_runner_smoke
            tests/test_bip_scan_runner_smoke.cpp)
        target_link_libraries(test_bip_scan_runner_smoke collider_core)
        add_test(NAME BipScanRunnerSmoke COMMAND test_bip_scan_runner_smoke
                 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(BipScanRunnerSmoke PROPERTIES
            SKIP_RETURN_CODE 77)
    endif()

    # v1.4.2 A.3: regression for kangaroo Tame init range-offset reduction.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_random_below.cpp")
        add_executable(test_random_below tests/test_random_below.cpp)
        add_test(NAME RandomBelow COMMAND test_random_below)
    endif()

    # v1.4.2 A.4: regression for kangaroo DP table full-256 keying.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_dp_table_full256.cpp")
        add_executable(test_dp_table_full256 tests/test_dp_table_full256.cpp)
        add_test(NAME DpTableFull256 COMMAND test_dp_table_full256)
    endif()

    # v1.4.2 A.7: regression for bech32 BIP-173 checksum verification.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bech32_checksum.cpp")
        add_executable(test_bech32_checksum tests/test_bech32_checksum.cpp)
        add_test(NAME Bech32Checksum COMMAND test_bech32_checksum)
    endif()

    # B1 wire-v4: bech32 P2WPKH ENCODER KAT (cross-checks Python server).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bech32_encode_kat.cpp")
        add_executable(test_bech32_encode_kat tests/test_bech32_encode_kat.cpp)
        add_test(NAME Bech32EncodeKat COMMAND test_bech32_encode_kat)
    endif()

    # B1 wire-v4: WIF decode/encode KAT (Wallet Import Format private keys).
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_wif_kat.cpp"
       AND TARGET collider_core)
        add_executable(test_wif_kat tests/test_wif_kat.cpp)
        target_link_libraries(test_wif_kat collider_core)
        add_test(NAME WifKat COMMAND test_wif_kat)
    endif()

    # B1 wire-v4: OpenSSL secp256k1 signer KAT for the AUTH frame.
    # Skipped when OpenSSL is unavailable; ctest treats exit 77 as SKIP.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_worker_identity_kat.cpp"
       AND TARGET collider_core AND OPENSSL_FOUND)
        add_executable(test_worker_identity_kat
            tests/test_worker_identity_kat.cpp)
        target_link_libraries(test_worker_identity_kat
            collider_core OpenSSL::SSL OpenSSL::Crypto)
        add_test(NAME WorkerIdentityKat COMMAND test_worker_identity_kat)
        set_tests_properties(WorkerIdentityKat PROPERTIES
            SKIP_RETURN_CODE 77)
    endif()

    # B1 wire-v4: canonical AUTH message bytes (C++ <-> Python).
    # Links collider_core for the session_log symbols pulled by
    # jlp_pool_client.cpp.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_auth_v4_canonical_kat.cpp"
       AND TARGET collider_pool AND TARGET collider_core)
        add_executable(test_jlp_auth_v4_canonical_kat
            tests/test_jlp_auth_v4_canonical_kat.cpp)
        target_link_libraries(test_jlp_auth_v4_canonical_kat
            collider_pool collider_core)
        add_test(NAME JlpAuthV4CanonicalKat
                 COMMAND test_jlp_auth_v4_canonical_kat)
    endif()

    # v1.4.2 B.6: regression for STATS_RSP NaN/Inf clamp + UB-safe cast.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_stats_sanitize.cpp")
        add_executable(test_stats_sanitize tests/test_stats_sanitize.cpp)
        add_test(NAME StatsSanitize COMMAND test_stats_sanitize)
    endif()

    # v1.4.2 Pool-B2: the dp_seq_window.hpp header and its test were deleted.
    # The header was documented as the canonical spec both client and server
    # must implement; in reality the C++ client never used it (the client is
    # a sender only) and the Python server uses a different scheme
    # (_dp_seq_high watermark + SEQ_REPLAY_WINDOW=1024 lookback). Dead code
    # pretending to close audit finding H4 was misleading -- the actual
    # replay-defence protocol is now documented in docs/jlp-protocol.md.

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

    # CLI parser + config-override matrix.
    # The test links the real parser (src/cli/cli_parser.cpp) directly so
    # parser bugs surface here. Previously the test re-implemented a
    # parse_args mirror in-test "because parse_args lives in main.cpp"; the
    # registry refactor (cli_parser.cpp + flag_spec.hpp) made that mirror
    # unnecessary. We compile cli_parser.cpp into the test target rather
    # than linking collider_core because collider_core pulls in detect_gpus,
    # run_pool_mode, brain_wallet_runner, etc., none of which the parser
    # needs. cli_parser.cpp itself has no library deps beyond Threads.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_cli_parser.cpp")
        add_executable(test_cli_parser
            tests/test_cli_parser.cpp
            src/cli/cli_parser.cpp
        )
        if(COLLIDER_PRO)
            target_compile_definitions(test_cli_parser PRIVATE COLLIDER_PRO=1)
        endif()
        target_link_libraries(test_cli_parser Threads::Threads)
        add_test(NAME CLIParser COMMAND test_cli_parser)
    endif()

    # v1.4.2 A-tier: generator-mode coverage. Walks StreamingBrainWallet
    # forward through PHASE_CYCLING -> RULE_STACKING -> ... -> MARKOV ->
    # PCFG -> KEYBOARD_WALK and asserts each new mode emits non-empty
    # candidate batches. Locks down the Phase 4 wiring against
    # accidental reversion to the prior stub-out.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_generator_modes.cpp" AND COLLIDER_PRO)
        add_executable(test_generator_modes tests/test_generator_modes.cpp)
        target_link_libraries(test_generator_modes collider_core Threads::Threads)
        target_include_directories(test_generator_modes PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME GeneratorModes COMMAND test_generator_modes)
    endif()

    # Repair R1: regression cover for restore_state() into a non-PHASE_CYCLING
    # iteration mode. Pre-R1 the mode-owned generator state (markov_matrix_,
    # pcfg_grammar_, stacked_rules_, hybrid_mask_plans_, keyboard_walk_seeds_)
    # was not re-initialized on resume, so next_candidate() returned "" until
    # the iteration budget burned out.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_resume_iteration_mode.cpp" AND COLLIDER_PRO)
        add_executable(test_resume_iteration_mode tests/test_resume_iteration_mode.cpp)
        target_link_libraries(test_resume_iteration_mode collider_core Threads::Threads)
        target_include_directories(test_resume_iteration_mode PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME ResumeIterationMode COMMAND test_resume_iteration_mode)
    endif()

    # --brute: BruteforceGenerator coverage. Charset, lex ordering,
    # multi-length sequencing, mid-run resume, batch-boundary stability.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bruteforce_generator.cpp" AND COLLIDER_PRO)
        add_executable(test_bruteforce_generator tests/test_bruteforce_generator.cpp)
        target_link_libraries(test_bruteforce_generator collider_core)
        target_include_directories(test_bruteforce_generator PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME BruteforceGenerator COMMAND test_bruteforce_generator)
    endif()

    # task #5: brute-mode state save/load round-trip via
    # BrainWalletSearchState. Pure CPU test, no GPU dependency.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_brute_resume_state.cpp" AND COLLIDER_PRO)
        add_executable(test_brute_resume_state tests/test_brute_resume_state.cpp)
        target_include_directories(test_brute_resume_state PRIVATE ${CMAKE_SOURCE_DIR})
        add_test(NAME BruteResumeState COMMAND test_brute_resume_state)
    endif()

    # v1.4.2 final-tier: multi-GPU concurrent dispatch regression for
    # BW-B3. Skips with exit 77 on single-GPU rigs (covers CI Linux/Mac).
    # The test is a .cu file and links CUDA::cudart, so it can only be
    # built when CUDA is the active backend. The Mac / Metal build path
    # has no equivalent test target; the Metal multi-GPU path is
    # exercised by the bench_metal_kangaroo runner instead.
    if(COLLIDER_PRO AND COLLIDER_USE_CUDA AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_multi_gpu_brain_wallet.cu" AND TARGET collider_core)
        add_executable(test_multi_gpu_brain_wallet tests/test_multi_gpu_brain_wallet.cu)
        set_target_properties(test_multi_gpu_brain_wallet PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY Shared
        )
        target_link_libraries(test_multi_gpu_brain_wallet
            collider_gpu collider_core CUDA::cudart Threads::Threads)
        add_test(NAME MultiGpuBrainWallet COMMAND test_multi_gpu_brain_wallet)
        # T3.15: gpu lock -- the multi-GPU dispatcher takes every visible
        # device; running concurrent ctest jobs against the same physical
        # GPU produces context conflicts and OOM races. RESOURCE_LOCK gpu
        # is the canonical ctest serialisation token; every test in this
        # file that touches a CUDA context shares the same lock so -j N
        # can still parallelise the host-only suites.
        set_tests_properties(MultiGpuBrainWallet PROPERTIES
            SKIP_RETURN_CODE 77
            RESOURCE_LOCK gpu)
    endif()

    # v1.4.2 quality: deterministic reproduction harness for the
    # minute-47 brainwallet CUDA fault (operator-reported regression).
    # Scenarios A/B/C exercise the large-to-small phase change, the
    # phase change plus hit-handling file I/O delay, and the cumulative
    # load path. RESOURCE_LOCK gpu serialises against other GPU tests;
    # TIMEOUT 1800 leaves headroom for scenario C (cumulative cycles).
    if(COLLIDER_PRO AND COLLIDER_USE_CUDA AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_phase_change_fault.cu" AND TARGET collider_core)
        add_executable(test_phase_change_fault tests/test_phase_change_fault.cu)
        set_target_properties(test_phase_change_fault PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY Shared
        )
        target_link_libraries(test_phase_change_fault
            collider_gpu collider_core CUDA::cudart Threads::Threads)
        add_test(NAME PhaseChangeFault COMMAND test_phase_change_fault)
        set_tests_properties(PhaseChangeFault PROPERTIES
            SKIP_RETURN_CODE 77
            RESOURCE_LOCK gpu
            TIMEOUT 1800)
    endif()

    # KAT for range_bits_from_be in src/core/byte_codec.hpp. Pins the
    # canonical Bitcoin-puzzle ranges (75/135/160) to expected
    # range_bits values so a regression in the 256-bit subtraction +
    # bit_length helper would fail loudly. Used by Phase 1.4
    # CudaRCKangarooBackend::initialize() to size the kangaroo solver.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_range_bits.cpp")
        add_executable(test_range_bits tests/test_range_bits.cpp)
        # Header-only target; byte_codec.hpp is inline-only.
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

    # P-B1 dx==0 SIMT guard fuzz test (test-quality T5). Pins the
    # algebraic correctness of the batch-inversion pre-process from
    # kangaroo_step_kernel_jlp / kangaroo_step_kernel_sota. Host-only:
    # mirrors the kernel's product-chain math against
    # collider::cpu::mod_mul / mod_inv. The kernel-driver complement
    # is test_kangaroo_dxz_kernel (T3.3, 2026-05-17) which exercises
    # the same guard via the actual CUDA dispatch path.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_kangaroo_dxz_fuzz.cpp")
        add_executable(test_kangaroo_dxz_fuzz tests/test_kangaroo_dxz_fuzz.cpp)
        add_test(NAME KangarooDxzFuzz COMMAND test_kangaroo_dxz_fuzz)
    endif()

    # T3.3 (2026-05-17). Kernel-driver complement to KangarooDxzFuzz.
    # Launches test_only_kangaroo_dxz_guard_kernel (defined in
    # src/gpu/kangaroo_kernel.cu) with crafted dx==0 inputs and
    # verifies the SIMT guard fires in the actual CUDA kernel binary
    # (the CPU mirror covers algebra; this covers the kernel path).
    # Skips via SKIP_RETURN_CODE 77 when no CUDA device is available.
    if(COLLIDER_USE_CUDA AND
       EXISTS "${CMAKE_SOURCE_DIR}/tests/test_kangaroo_dxz_kernel.cu" AND
       TARGET collider_core AND TARGET collider_gpu)
        add_executable(test_kangaroo_dxz_kernel tests/test_kangaroo_dxz_kernel.cu)
        set_target_properties(test_kangaroo_dxz_kernel PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY Shared
        )
        target_link_libraries(test_kangaroo_dxz_kernel
            collider_gpu collider_core CUDA::cudart)
        add_test(NAME KangarooDxzKernel COMMAND test_kangaroo_dxz_kernel)
        # gpu resource lock + skip code: same pattern as the
        # MultiGpuBrainWallet / PuzzleOptimizedInv tests above.
        set_tests_properties(KangarooDxzKernel PROPERTIES
            SKIP_RETURN_CODE 77
            RESOURCE_LOCK gpu)
    endif()

    # v1.5 asymmetric kangaroo mode self-test. Drives RCGpuKang directly
    # (NOT via rckangaroo_wrapper, which is a process singleton) in two
    # sequential passes: TAME_ONLY and WILD_ONLY. The host captures every
    # emitted DP, then simulates the pool server's collision search and
    # verifies cross-instance DPs recover a known private key via the
    # SOTA formula. Pins the v1.5 plan's algorithm-level theft resistance:
    # neither pass alone has the data to compute the key.
    #
    # Links ONLY against rckangaroo + CUDA::cudart (NOT collider_gpu),
    # since the wrapper's own AddPointsToList / gGenMode / gTotalErrors
    # singletons would conflict with the test's local versions.
    if(COLLIDER_USE_CUDA AND COLLIDER_USE_RCKANGAROO AND TARGET rckangaroo AND
       EXISTS "${CMAKE_SOURCE_DIR}/tests/test_kangaroo_mode_asymmetric.cu")
        add_executable(test_kangaroo_mode_asymmetric
            tests/test_kangaroo_mode_asymmetric.cu)
        # rckangaroo is a CUDA-separable static lib whose RCGpuCore.cu
        # defines __constant__ symbols (jmp2_table) referenced from
        # cudaMemcpyToSymbol on the host. The device-side relocation
        # that resolves those symbols at runtime is finalized in the
        # final executable's device-link step, which only happens when
        # the target itself is marked CUDA-separable. Without these
        # properties cudaMemcpyToSymbol returns cudaErrorInvalidSymbol
        # ("named symbol not found") and Prepare() fails for every
        # mode. Same pattern used by test_kangaroo_dxz_kernel above.
        # Match test_kangaroo_dxz_kernel above: CUDA_SEPARABLE_COMPILATION
        # ON. The executable's device link absorbs rckangaroo's device
        # side and registers the __constant__ jmp2_table symbol in this
        # executable's fatbin so cudaMemcpyToSymbol can find it at
        # runtime. rckangaroo's CMakeLists.txt drops
        # CUDA_RESOLVE_DEVICE_SYMBOLS so the device link defers to here.
        #
        # /WHOLEARCHIVE forces every .obj from rckangaroo.lib into the
        # exe. Without it, the linker pulls only obj files whose host
        # symbols are referenced -- which IS RCGpuCore.cu.obj (via
        # cuSetGpuParams) -- but then COMDAT-level deadstrip can drop
        # __sti____cudaRegisterAll (the static init that calls
        # __cudaRegisterVar for jmp2_table). With the static init
        # dropped, cudaMemcpyToSymbol(jmp2_table, ...) returns
        # cudaErrorInvalidSymbol at runtime. WHOLEARCHIVE keeps every
        # init COMDAT alive.
        set_target_properties(test_kangaroo_mode_asymmetric PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY Shared)
        target_link_libraries(test_kangaroo_mode_asymmetric
            rckangaroo CUDA::cudart)
        if(MSVC)
            target_link_options(test_kangaroo_mode_asymmetric PRIVATE
                "LINKER:/WHOLEARCHIVE:rckangaroo")
        else()
            target_link_options(test_kangaroo_mode_asymmetric PRIVATE
                "LINKER:--whole-archive"
                "LINKER:-lrckangaroo"
                "LINKER:--no-whole-archive")
        endif()
        add_test(NAME KangarooModeAsymmetric
            COMMAND test_kangaroo_mode_asymmetric)
        set_tests_properties(KangarooModeAsymmetric PROPERTIES
            SKIP_RETURN_CODE 77
            RESOURCE_LOCK gpu
            TIMEOUT 120)
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
        # T3.15: gpu lock -- this test runs CUDA kernels when a device
        # is present. See MultiGpuBrainWallet for the lock rationale.
        set_tests_properties(BrainWalletV2 PROPERTIES
            SKIP_RETURN_CODE 77
            RESOURCE_LOCK gpu)
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
    # under #if COLLIDER_USE_CUDA; the host code paths are dead in the
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
    # Pure C++. Header-only SHA-256 + RIPEMD-160 + hash160 reference,
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

    # task F: Keccak-256 CPU reference KATs. Ethereum-flavor (0x01
    # padding byte, NOT FIPS-202 SHA-3-256's 0x06). Validates the host-side
    # reference that the device kernel mirrors.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_keccak256_cpu.cpp" AND COLLIDER_PRO)
        add_executable(test_keccak256_cpu tests/v2/test_keccak256_cpu.cpp)
        target_include_directories(test_keccak256_cpu PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME Keccak256Cpu COMMAND test_keccak256_cpu)
    endif()

    # task F stage 2: device Keccak-256 vs CPU reference parity.
    if(COLLIDER_BACKEND STREQUAL "CUDA" AND COLLIDER_PRO
       AND EXISTS "${CMAKE_SOURCE_DIR}/tests/v2/test_keccak256_device.cu")
        add_executable(test_keccak256_device tests/v2/test_keccak256_device.cu)
        set_target_properties(test_keccak256_device PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_include_directories(test_keccak256_device PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME Keccak256Device COMMAND test_keccak256_device)
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
    # successful remote validation. No filesystem / network; the test
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
    # No sockets, no threads; pure logic against the collider_pool library.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_manager.cpp" AND TARGET collider_pool)
        add_executable(test_jlp_pool_manager tests/test_jlp_pool_manager.cpp)
        # v1.4.2 quality: collider_pool now depends on the session_log
        # symbols in collider_core (milestone/update_session_state/
        # flush_session_state -- wired into the connect / AUTH / WORK_ASN
        # / DP submit / disconnect paths in jlp_pool_client.cpp). The pool
        # tests must therefore link collider_core too. Listing it AFTER
        # collider_pool keeps the dependency-flow linker happy (consumers
        # before producers on the gcc/clang link line; MSVC ignores the
        # order).
        target_link_libraries(test_jlp_pool_manager
            collider_pool collider_core Threads::Threads)
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
        # See JLPPoolManager above: collider_core supplies the
        # session_log symbols collider_pool now references.
        target_link_libraries(test_jlp_pool_protocol
            collider_pool collider_core Threads::Threads)
        if(WIN32)
            # ws2_32 for the test's own mock-server WinSock calls (the
            # collider_pool library transitively links it too, but listing it
            # here keeps the dependency explicit).
            target_link_libraries(test_jlp_pool_protocol ws2_32)
        endif()
        add_test(NAME JLPPoolProtocol COMMAND test_jlp_pool_protocol)
        # The protocol suite uses real threads + sockets; give it room.
        # T3.15: RESOURCE_LOCK pool_port serialises this with the other
        # JLP mock-server suites. Each one stands up an in-process listener
        # on a fixed 127.0.0.1 port; parallel ctest invocations would race
        # on the same port and fail with "address in use".
        set_tests_properties(JLPPoolProtocol PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK pool_port)
    endif()

    # JLP pool reconnect / disconnect handling tests. Mock TCP server, no TLS.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_reconnect.cpp" AND TARGET collider_pool)
        add_executable(test_jlp_pool_reconnect tests/test_jlp_pool_reconnect.cpp)
        # See JLPPoolManager above: collider_core supplies the
        # session_log symbols collider_pool now references.
        target_link_libraries(test_jlp_pool_reconnect
            collider_pool collider_core Threads::Threads)
        if(WIN32)
            target_link_libraries(test_jlp_pool_reconnect ws2_32)
        endif()
        add_test(NAME JLPPoolReconnect COMMAND test_jlp_pool_reconnect)
        # T3.15: pool_port lock -- see JLPPoolProtocol above.
        set_tests_properties(JLPPoolReconnect PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK pool_port)
    endif()

    # S1 regression: WORK_ASN dp_bits validation. Drives a mock server that
    # sends out-of-range dp_bits values (255, 7, 33) and asserts the client
    # rejects the assignment + disconnects without ever firing the work
    # callback. Boundary values 8 and 32 must still pass.
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_jlp_pool_dp_bits_validation.cpp"
       AND TARGET collider_pool)
        add_executable(test_jlp_pool_dp_bits_validation
            tests/test_jlp_pool_dp_bits_validation.cpp)
        # See JLPPoolManager above: collider_core supplies the
        # session_log symbols collider_pool now references.
        target_link_libraries(test_jlp_pool_dp_bits_validation
            collider_pool collider_core Threads::Threads)
        if(WIN32)
            target_link_libraries(test_jlp_pool_dp_bits_validation ws2_32)
        endif()
        add_test(NAME JLPPoolDpBitsValidation
                 COMMAND test_jlp_pool_dp_bits_validation)
        # T3.15: pool_port lock -- see JLPPoolProtocol above.
        set_tests_properties(JLPPoolDpBitsValidation PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK pool_port)
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
            target_link_libraries(test_secp256k1_inv collider_gpu collider_core CUDA::cudart)
            add_test(NAME Secp256k1Inv COMMAND test_secp256k1_inv)
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(Secp256k1Inv PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        # Device HMAC-SHA512 KAT (RFC 4231 TC1/TC2/TC4/TC6). Pins the
        # output of device::hmac_sha512 in src/gpu/v2/device_hashes.cuh
        # against the published vectors; catches regressions in the
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
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(DeviceHmacSha512KAT PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_ec_table_consistency.cu")
            add_executable(test_ec_table_consistency tests/test_ec_table_consistency.cu)
            set_target_properties(test_ec_table_consistency PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_ec_table_consistency collider_gpu collider_core CUDA::cudart)
            add_test(NAME EcTableConsistency COMMAND test_ec_table_consistency)
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(EcTableConsistency PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
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
            target_link_libraries(test_puzzle_optimized_inv collider_gpu collider_core CUDA::cudart)
            add_test(NAME PuzzleOptimizedInv COMMAND test_puzzle_optimized_inv)
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(PuzzleOptimizedInv PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
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
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(GpuHash160 PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        # TP-5: fused-pipeline boundary KAT (count=0, count=1,
        # MAX_MATCHES_PER_BATCH saturation with OOB sentinel).
        if(COLLIDER_PRO AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_fused_pipeline_oob.cu" AND TARGET collider_core)
            add_executable(test_fused_pipeline_oob tests/test_fused_pipeline_oob.cu)
            set_target_properties(test_fused_pipeline_oob PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_fused_pipeline_oob collider_gpu collider_core CUDA::cudart)
            add_test(NAME FusedPipelineOob COMMAND test_fused_pipeline_oob)
            set_tests_properties(FusedPipelineOob PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu
                TIMEOUT 60)
        endif()

        # GpuMultiAddr exercises the V2 multi_addr=true path of the fused
        # brain-wallet kernel (the path the user's --brainwallet-v2 --resume
        # session was on when it took the cudaErrorIllegalAddress fault).
        # Insert mode rotates compressed P2PKH / uncompressed P2PKH /
        # P2SH-P2WPKH so the kernel's fused_multiaddr_extra_check sub-paths
        # are all exercised, and runs against both a pow-of-2 bloom (mask
        # fast path) and a prime-size bloom (modulo path, same code as the
        # production 2.57 GB bloom).
        if(COLLIDER_PRO AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_gpu_multi_addr.cu" AND TARGET collider_core)
            add_executable(test_gpu_multi_addr tests/test_gpu_multi_addr.cu)
            set_target_properties(test_gpu_multi_addr PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_gpu_multi_addr collider_gpu collider_core CUDA::cudart)
            add_test(NAME GpuMultiAddr COMMAND test_gpu_multi_addr)
            set_tests_properties(GpuMultiAddr PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
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
            target_link_libraries(test_ec_mul_known_answers collider_gpu collider_core CUDA::cudart)
            add_test(NAME EcMulKnownAnswers COMMAND test_ec_mul_known_answers)
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(EcMulKnownAnswers PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        # T3.1 (A-tier wave 1, 2026-05-17): regression KATs that pin the
        # crypto-correctness landmines deleted in T0.3.a/b/c. Each test
        # acts as a reintroduction trap for a deleted symbol; if the
        # kernel is brought back, the test runs and asserts the
        # reintroduced kernel passes the original KAT (or fails loudly
        # if the contributor leaves the stub body in place). See the
        # per-test comment block for the bug each one guards.

        # T0.3.a guard: secp256k1_batch_mul (windowed batch-inversion
        # variant, double 2^(5w) shift).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_secp256k1_batch_mul_kat.cu")
            add_executable(test_secp256k1_batch_mul_kat
                tests/test_secp256k1_batch_mul_kat.cu)
            set_target_properties(test_secp256k1_batch_mul_kat PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_secp256k1_batch_mul_kat
                collider_gpu collider_core CUDA::cudart)
            add_test(NAME Secp256k1BatchMulKAT
                     COMMAND test_secp256k1_batch_mul_kat)
            set_tests_properties(Secp256k1BatchMulKAT PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        # T0.3.b absence trap: mega_fused_brainwallet_batch (orphaned
        # persistent kernel that carried the same windowed-mul double-shift
        # bug as the deleted ec_mul_batch_optimized_kernel). The previous
        # registration shipped an always-pass stub; reauthored 2026-05-17 to
        # actually scan the compiled collider_gpu.lib archive for the
        # deleted symbol substrings. The test fails if any of them are
        # found, catching accidental reintroduction via copy-paste, stale
        # object file, or source-tree re-add.
        #
        # COLLIDER_GPU_LIB_PATH is injected as a compile-time string so the
        # test can find the archive at runtime without depending on the
        # working directory ctest happens to launch from. The test still
        # has no link-time dependency on collider_gpu itself; the lib is
        # only inspected as bytes.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_mega_fused_brainwallet_kat.cu")
            add_executable(test_mega_fused_brainwallet_kat
                tests/test_mega_fused_brainwallet_kat.cu)
            set_target_properties(test_mega_fused_brainwallet_kat PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_compile_definitions(test_mega_fused_brainwallet_kat PRIVATE
                COLLIDER_GPU_LIB_PATH="$<TARGET_FILE:collider_gpu>")
            add_dependencies(test_mega_fused_brainwallet_kat collider_gpu)
            add_test(NAME MegaFusedBrainwalletAbsence
                     COMMAND test_mega_fused_brainwallet_kat)
            set_tests_properties(MegaFusedBrainwalletAbsence PROPERTIES
                SKIP_RETURN_CODE 77)
        endif()

        # T0.3.c guard: sha256_batch (single-block sha256_hash that
        # truncated inputs > 55 bytes). Unlike the two reintroduction
        # traps above, sha256_batch is alive in production after the T0.3.c
        # rewire to sha256_hash_long; this KAT pins the boundary so
        # any future "fast path" cap at 55 fails on inputs of 56/64/100.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_sha256_batch_kat.cu")
            add_executable(test_sha256_batch_kat
                tests/test_sha256_batch_kat.cu)
            set_target_properties(test_sha256_batch_kat PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_sha256_batch_kat
                collider_gpu collider_core CUDA::cudart)
            add_test(NAME Sha256BatchKAT
                     COMMAND test_sha256_batch_kat)
            set_tests_properties(Sha256BatchKAT PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
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
            target_link_libraries(test_kangaroo_small_puzzle collider_gpu collider_core CUDA::cudart)
            add_test(NAME KangarooSmallPuzzle COMMAND test_kangaroo_small_puzzle)
            # T3.15: gpu lock -- see MultiGpuBrainWallet.
            set_tests_properties(KangarooSmallPuzzle PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        # v1.4.2 C.1: KAT for the shared GLV decomposition header
        # (src/gpu/glv_decompose.cuh). Self-contained; does not depend on
        # collider_gpu (only includes the header), so a header bug fails
        # this test directly without requiring the full puzzle/brainwallet
        # pipeline. Verifies for each test scalar k: (a) Babai bound
        # k1[3] == k2[3] == 0, (b) algebraic identity
        # signed_k1 + signed_k2 * lambda == k (mod n).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_glv_decompose.cu")
            add_executable(test_glv_decompose tests/test_glv_decompose.cu)
            set_target_properties(test_glv_decompose PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Shared
            )
            target_link_libraries(test_glv_decompose CUDA::cudart)
            add_test(NAME GlvDecompose COMMAND test_glv_decompose)
            # T3.15: gpu lock -- GLV decompose runs a CUDA kernel even
            # though the math is small; share the same lock as the rest.
            set_tests_properties(GlvDecompose PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()
    endif()

    # ========================================================================
    # brainwallet TUI overhaul: pre-registered test targets
    # ========================================================================
    # Wave 1 builder-foundation registers every test the plan introduces so
    # downstream phase builders only fill in the .cpp content. Each stub
    # ships as `int main(){return 0;}` so ctest is green throughout.
    # PRO-only because every test exercises code that lives in collider_core
    # under #if COLLIDER_PRO.
    if(COLLIDER_PRO)
        # Phase 1 (builder-threading: scan-state-atomics).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_scan_state_atomics.cpp"
           AND TARGET collider_core)
            add_executable(test_scan_state_atomics tests/test_scan_state_atomics.cpp)
            target_link_libraries(test_scan_state_atomics collider_core)
            add_test(NAME ScanStateAtomics COMMAND test_scan_state_atomics)
        endif()

        # Phase 1 (builder-panels: baseline-panels): FTXUI headless render.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_tui_panels.cpp"
           AND TARGET collider_core)
            add_executable(test_tui_panels tests/test_tui_panels.cpp)
            target_link_libraries(test_tui_panels collider_core)
            add_test(NAME TuiPanels COMMAND test_tui_panels)
        endif()

        # Phase 1 (builder-panels: baseline-panels): Unicode + ASCII sparkline.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_sparkline.cpp"
           AND TARGET collider_core)
            add_executable(test_sparkline tests/test_sparkline.cpp)
            target_link_libraries(test_sparkline collider_core)
            add_test(NAME Sparkline COMMAND test_sparkline)
        endif()

        # Phase 1 (builder-threading: cooked-mode-guard): fork-child SIGINT.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_cooked_mode_guard.cpp"
           AND TARGET collider_core)
            add_executable(test_cooked_mode_guard tests/test_cooked_mode_guard.cpp)
            target_link_libraries(test_cooked_mode_guard collider_core)
            add_test(NAME CookedModeGuard COMMAND test_cooked_mode_guard)
        endif()

        # T1-H: StdioCapture nesting safety. Header-only target; no
        # link against collider_core required.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_stdio_capture_nesting.cpp")
            add_executable(test_stdio_capture_nesting
                tests/test_stdio_capture_nesting.cpp)
            target_include_directories(test_stdio_capture_nesting
                PRIVATE ${CMAKE_SOURCE_DIR}/src)
            add_test(NAME StdioCaptureNesting
                COMMAND test_stdio_capture_nesting)
        endif()

        # Phase 2 (builder-gpu: nvml-wrapper): happy-path + degraded.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_nvml_wrapper.cpp"
           AND TARGET collider_core)
            add_executable(test_nvml_wrapper tests/test_nvml_wrapper.cpp)
            target_link_libraries(test_nvml_wrapper collider_core)
            if(COLLIDER_BACKEND STREQUAL "CUDA")
                target_link_libraries(test_nvml_wrapper CUDA::cudart)
            endif()
            add_test(NAME NvmlWrapper COMMAND test_nvml_wrapper)
            # T3.15: gpu lock -- NVML init shares the driver context with
            # CUDA tests; serialise to avoid driver-init races.
            set_tests_properties(NvmlWrapper PROPERTIES
                SKIP_RETURN_CODE 77
                RESOURCE_LOCK gpu)
        endif()

        # Phase 3 (builder-perf: cuda-event-instrumentation): synthetic stream.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_perf_instrumentation.cpp"
           AND TARGET collider_core)
            add_executable(test_perf_instrumentation tests/test_perf_instrumentation.cpp)
            target_link_libraries(test_perf_instrumentation collider_core)
            # Q7: perf_instrumentation.cpp now registers a gpu-side adapter
            # via collider::gpu::instrumentation::register_callbacks, whose
            # storage lives in collider_gpu (instrumentation_hook.cpp). The
            # test must link collider_gpu to resolve that symbol.
            if(COLLIDER_BACKEND STREQUAL "CUDA" AND TARGET collider_gpu)
                target_link_libraries(test_perf_instrumentation collider_gpu)
            endif()
            if(COLLIDER_BACKEND STREQUAL "CUDA")
                target_link_libraries(test_perf_instrumentation CUDA::cudart)
            endif()
            add_test(NAME PerfInstrumentation COMMAND test_perf_instrumentation)
        endif()

        # Phase 4 (builder-threading: runtime-control-input).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_input_handler.cpp"
           AND TARGET collider_core)
            add_executable(test_input_handler tests/test_input_handler.cpp)
            target_link_libraries(test_input_handler collider_core)
            add_test(NAME InputHandler COMMAND test_input_handler)
        endif()

        # Phase 4 (builder-threading: runtime-yaml): round-trip load/save.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_runtime_yaml.cpp"
           AND TARGET collider_core)
            add_executable(test_runtime_yaml tests/test_runtime_yaml.cpp)
            target_link_libraries(test_runtime_yaml collider_core)
            add_test(NAME RuntimeYaml COMMAND test_runtime_yaml)
        endif()

        # Phase 5 (builder-threading: streaming-generator-hot-swap).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_wordlist_hot_swap.cpp"
           AND TARGET collider_core)
            add_executable(test_wordlist_hot_swap tests/test_wordlist_hot_swap.cpp)
            target_link_libraries(test_wordlist_hot_swap collider_core)
            add_test(NAME WordlistHotSwap COMMAND test_wordlist_hot_swap)
        endif()

        # TP-2: WordlistComposer recombine action end-to-end test.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_wordlist_composer.cpp"
           AND TARGET collider_core)
            add_executable(test_wordlist_composer tests/test_wordlist_composer.cpp)
            target_link_libraries(test_wordlist_composer collider_core)
            add_test(NAME WordlistComposer COMMAND test_wordlist_composer)
            set_tests_properties(WordlistComposer PROPERTIES TIMEOUT 60)
        endif()

        # TP-3: BIP-32 KAT against the standard spec test vectors.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bip32_kat.cpp"
           AND TARGET collider_core)
            add_executable(test_bip32_kat tests/test_bip32_kat.cpp)
            target_link_libraries(test_bip32_kat collider_core)
            add_test(NAME Bip32Kat COMMAND test_bip32_kat)
            set_tests_properties(Bip32Kat PROPERTIES TIMEOUT 30)
        endif()

        # TP-6: atomic_write_state_file durability KAT (empty / large
        # / overwrite / leftover-tmp recovery / 1000-iter stress).
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_state_atomic_write.cpp"
           AND TARGET collider_core)
            add_executable(test_state_atomic_write tests/test_state_atomic_write.cpp)
            target_link_libraries(test_state_atomic_write collider_core)
            add_test(NAME StateAtomicWrite COMMAND test_state_atomic_write)
            set_tests_properties(StateAtomicWrite PROPERTIES TIMEOUT 60)
        endif()

        # TP-11: SettingsState concurrent edit + poll atomicity stress.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_settings_atomicity.cpp"
           AND TARGET collider_core)
            add_executable(test_settings_atomicity tests/test_settings_atomicity.cpp)
            target_link_libraries(test_settings_atomicity collider_core)
            add_test(NAME SettingsAtomicity COMMAND test_settings_atomicity)
            set_tests_properties(SettingsAtomicity PROPERTIES TIMEOUT 30)
        endif()

        # TP-8: generator_candidates_per_iteration budget enforcement.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_generator_budget.cpp"
           AND TARGET collider_core)
            add_executable(test_generator_budget tests/test_generator_budget.cpp)
            target_link_libraries(test_generator_budget collider_core)
            add_test(NAME GeneratorBudget COMMAND test_generator_budget)
            set_tests_properties(GeneratorBudget PROPERTIES TIMEOUT 30)
        endif()

        # TP-9: CLI > config.yml precedence + auto-enable side effects.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_cli_config_precedence.cpp"
           AND TARGET collider_core)
            add_executable(test_cli_config_precedence tests/test_cli_config_precedence.cpp)
            target_link_libraries(test_cli_config_precedence collider_core)
            add_test(NAME CliConfigPrecedence COMMAND test_cli_config_precedence)
            set_tests_properties(CliConfigPrecedence PROPERTIES TIMEOUT 10)
        endif()

        # Phase 5 (builder-threading: plugin-runner): synthetic echo plugin.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_plugin_runner.cpp"
           AND TARGET collider_core)
            add_executable(test_plugin_runner tests/test_plugin_runner.cpp)
            target_link_libraries(test_plugin_runner collider_core)
            add_test(NAME PluginRunner COMMAND test_plugin_runner)
        endif()

        # Tier 1 perf F-log (builder-perf): EmptyHitWriter steady-state,
        # overflow drop-oldest, and concurrent stop+enqueue scenarios.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_empty_hit_writer.cpp"
           AND TARGET collider_core)
            add_executable(test_empty_hit_writer tests/test_empty_hit_writer.cpp)
            target_link_libraries(test_empty_hit_writer collider_core Threads::Threads)
            add_test(NAME EmptyHitWriter COMMAND test_empty_hit_writer)
        endif()

        # 2026-05-17 reviewer coverage: redact_argv golden cases, the
        # SessionState JSON shape, shutdown_session_log double-fire
        # guard, prune_old_logs at the kSessionLogRetainCount boundary,
        # and the atomic_write_state_file tmp+rename + unwind contract.
        # Routes through collider::log::detail trampolines so the test
        # exec does not need anonymous-namespace internals.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_session_log.cpp"
           AND TARGET collider_core)
            add_executable(test_session_log tests/test_session_log.cpp)
            target_link_libraries(test_session_log collider_core)
            add_test(NAME SessionLog COMMAND test_session_log)
        endif()

        # Bloom false-positive-rate KAT (Tier 4 T7). Pins the empirical FP
        # rate of UTXOBloomBuilder at 1e-5 (loose GPU bloom) and 1e-7
        # (tight CPU bloom) within a 2x band. Header-only against
        # src/tools/utxo_bloom_builder.hpp; no link deps beyond the C++
        # runtime and bech32. Runs in both Free and Pro builds; the bloom
        # math is edition-agnostic.
        if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_bloom_fp_rate.cpp")
            add_executable(test_bloom_fp_rate tests/test_bloom_fp_rate.cpp)
            add_test(NAME BloomFpRate COMMAND test_bloom_fp_rate)
            # 10M MurmurHash3-128 probes per trial; ~3-5s on desktop CPU,
            # generous timeout for slower CI runners.
            set_tests_properties(BloomFpRate PROPERTIES TIMEOUT 120)
        endif()
    endif()
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(COLLIDER_BUILD_BENCHMARKS AND COLLIDER_BACKEND STREQUAL "CUDA")
    # Standalone full-pipeline benchmark. Pro-only because the bench core
    # depends on the fused brain-wallet kernels in collider_gpu (free
    # build's GPU library does not include them) and on the PerfCollector
    # symbols in collider_core that fused_pipeline.cu references.
    if(COLLIDER_PRO AND EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/bench_gpu_pipeline.cu"
       AND TARGET collider_core AND TARGET collider_gpu)
        add_executable(bench_gpu_pipeline benchmarks/bench_gpu_pipeline.cu)
        set_target_properties(bench_gpu_pipeline PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RUNTIME_LIBRARY Shared
        )
        target_link_libraries(bench_gpu_pipeline
            collider_core
            collider_gpu
            CUDA::cudart
        )
    endif()

    # v1.4.1 BW-DEDUP-3: bench_bloom_ab.cu was used to pick between
    # the two bloom impls. After the verdict (B-texture wins), the
    # losing impl was deleted and the bench retired; its purpose
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

if(TARGET generate_worker_key)
    install(TARGETS generate_worker_key RUNTIME DESTINATION bin)
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
