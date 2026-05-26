# CMakeLists.txt for RCKangaroo integration into theCollider.
# Original software: (c) 2024, RetiredCoder (RC), licensed under GPLv3.
#
# ============================================================================
# Modifications by SixCyber LLC, licensed under GPLv3 per the
# original work. See ../../LICENSE at the repository root and
# ../../THIRD_PARTY_LICENSES.md for the project-wide dependency inventory.
#
# Modification history for this file:
#   2026-05-21 (v1.5.0):
#     - CUDA_RESOLVE_DEVICE_SYMBOLS dropped to OFF so the static library
#       defers the device-link step to the executable target. This is
#       required for direct test consumers of rckangaroo (such as
#       test_kangaroo_mode_asymmetric in the top-level CMakeLists.txt)
#       to register the __constant__ jmp2_table symbol correctly via
#       cudaMemcpyToSymbol. See the inline comments at the
#       CUDA_RESOLVE_DEVICE_SYMBOLS property below.
#   Earlier:
#     - Wrapped the build as a static library named `rckangaroo` for
#       integration into theCollider via target_link_libraries instead
#       of building a standalone executable as upstream does.
# ============================================================================

cmake_minimum_required(VERSION 3.20)

# RCKangaroo library (built as part of thePuzzler)
set(RCKANGAROO_SOURCES
    Ec.cpp
    GpuKang.cpp
    RCGpuCore.cu
    utils.cpp
)

# Create library (not executable - we integrate it into thePuzzler)
add_library(rckangaroo STATIC ${RCKANGAROO_SOURCES})

# CUDA settings.
#
# CUDA_RESOLVE_DEVICE_SYMBOLS was previously ON, which made rckangaroo
# do its own device-link step and produce a self-contained fatbin
# inside the static archive. That worked for the main `collider` /
# `collider_pro` executables (which double-device-link anyway via
# their own CUDA_RESOLVE_DEVICE_SYMBOLS ON), but it broke direct test
# consumers: cudaMemcpyToSymbol(jmp2_table, ...) returned
# cudaErrorInvalidSymbol because the test executable's CUDA runtime
# couldn't find the __constant__ symbol registered through
# rckangaroo's internal fatbin.
#
# Per CMake's documented pattern for CUDA static libs that participate
# in further linking, the device-link step is deferred to consumers
# (every consumer in this project sets CUDA_RESOLVE_DEVICE_SYMBOLS ON
# at the executable level: see CMakeLists.txt near `collider` exe and
# the new test_kangaroo_mode_asymmetric target). Removing
# CUDA_RESOLVE_DEVICE_SYMBOLS here lets consumers do exactly one
# device-link of rckangaroo's device side instead of two, which is
# what the runtime expects.
set_target_properties(rckangaroo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(rckangaroo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA runtime
target_link_libraries(rckangaroo
    CUDA::cudart
)

# Compile flags - use generator expressions to separate CXX and CUDA flags
if(MSVC)
    target_compile_options(rckangaroo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
    )
else()
    target_compile_options(rckangaroo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-unused-function>
    )
endif()

# CUDA architectures - match parent project
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET rckangaroo PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()

# CUDA optimization flags
target_compile_options(rckangaroo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
