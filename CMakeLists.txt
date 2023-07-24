# Source: https://www.reddit.com/r/ROCm/comments/12bmygw/how_do_you_build_apps_with_hipblas_using_cmake/
cmake_minimum_required(VERSION 3.16)

project(hip_example VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")

# Find Modules
set(CMAKE_MODULE_PATH "/opt/rocm/hip/cmake")
find_package(HIP MODULE REQUIRED)

# Set compilation standard
set(CMAKE_CXX_COMPILER "/opt/rocm/hip/bin/hipcc")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If nvidia platform
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "HIP_PLATFORM: nvidia")
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES "70;75")
else()
  message(STATUS "HIP_PLATFORM: amd")
  set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908")
endif()

# Test programs
add_executable(hip_example main.cpp)
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "link to CUDA::cudart")
  target_compile_definitions(hip_example PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_link_libraries(hip_example PRIVATE CUDA::cudart)
else()
  message(STATUS "link to hip::device")
  target_compile_definitions(hip_example PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(hip_example PRIVATE hip::device)
endif()