# Source: https://www.reddit.com/r/ROCm/comments/12bmygw/how_do_you_build_apps_with_hipblas_using_cmake/
cmake_minimum_required(VERSION 3.16)

project(hip_example VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "Debug")

# Find Modules
if (NOT DEFINED ROCM_PATH)
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

if (DEFINED ENV{GPU_PLATFORM})
    set(HIP_PLATFORM "$ENV{GPU_PLATFORM}")
elseif (DEFINED ENV{NVIDIA_DRIVER_CAPABILITIES})
    set(HIP_PLATFORM "nvidia")
endif()

# Find hip
if(${HIP_PLATFORM} MATCHES "nvidia")
  set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake")
  find_package(HIP MODULE REQUIRED)
else()
  find_package(hip)
endif()

# Set compilation standard
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If nvidia platform
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "HIP_PLATFORM: nvidia")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES "70;75")
endif()

# Basic test programm
add_executable(hip_example example.cpp)
message("env is ${HIP_PLATFORM}")
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "link to CUDA::cudart")
  target_compile_definitions(hip_example PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_link_libraries(hip_example PRIVATE CUDA::cudart)
else()
  message(STATUS "link to hip::device")
  target_link_libraries(hip_example hip::host)
endif()

# Memmory mapped file test
add_executable(memory memory.cpp)
message("env is ${HIP_PLATFORM}")
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "link to CUDA::cudart")
  target_compile_definitions(memory PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_link_libraries(memory PRIVATE CUDA::cudart)
else()
  message(STATUS "link to hip::device")
  target_link_libraries(memory hip::host)
endif()
