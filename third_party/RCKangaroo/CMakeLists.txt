# CMakeLists.txt for RCKangaroo integration into thePuzzler
# Original software: (c) 2024, RetiredCoder (RC)
# License: GPLv3

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

# CUDA settings
set_target_properties(rckangaroo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
