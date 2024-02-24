# Inspired by: https://www.reddit.com/r/ROCm/comments/12bmygw/how_do_you_build_apps_with_hipblas_using_cmake/
cmake_minimum_required(VERSION 3.21.7)

project(hip_example VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_MODULE_PATH "/opt/rocm-6.0.0/lib/cmake/hip")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

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
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/opt/rocm/hip/cmake")
  find_package(HIP MODULE REQUIRED)
else()
  find_package(hip)
endif()

# Can be used with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Set compilation standard
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If nvidia platform
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "HIP_PLATFORM: nvidia")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES "70;75")
endif()

# Checks timeout and latency
add_executable(checks checks.cpp)
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "link to CUDA::cudart")
  set(CMAKE_EXE_LINKER_FLAGS "")
  target_compile_definitions(checks PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
  message(STATUS "link to hip::device")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fgpu-rdc -Wno-deprecated-register")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fgpu-rdc -Wno-deprecated-register")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fgpu-rdc --hip-link")
  target_link_libraries(checks hip::host)
endif()

# Basic example
add_executable(hip_example example.cpp)
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "link to CUDA::cudart")
  set(CMAKE_EXE_LINKER_FLAGS "")
  target_compile_definitions(hip_example PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
  message(STATUS "link to hip::device")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fgpu-rdc -Wno-deprecated-register")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fgpu-rdc -Wno-deprecated-register")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fgpu-rdc --hip-link")
  target_link_libraries(hip_example hip::host)
endif()

# Memmory mapped file test
add_executable(memory memory.cpp)
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "link to CUDA::cudart")
  set(CMAKE_EXE_LINKER_FLAGS "")
  target_compile_definitions(memory PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
  message(STATUS "link to hip::device")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fgpu-rdc -Wno-deprecated-register")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fgpu-rdc -Wno-deprecated-register")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fgpu-rdc --hip-link")
  target_link_libraries(memory hip::host)
endif()