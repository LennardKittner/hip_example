# hip_example

A simple hip example programm.

## Manuel Building

```
hipcc -o hip_example main.cpp
```

### Building With Cmake

If you are using a nvidia gpu you may have to first set the `GPU_PLATFORM`
```
export GPU_PLATFORM=nvidia
```
Then build with
```
mkdrid build
cd build
cmake ..
```
or
```
rm -rf build && cmake -B build && cmake --build build && ./build/hip_example
```

The environment variables in the environment_nvidia file can help to solve some issues but you should first try to build **without** them.

## Useful Information About HIP

## Introduction
https://www.youtube.com/watch?v=3ZXbRJVvgJs

## Atomics
https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#atomic-functions

## Synchronization
https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#warp-cross-lane-functions

https://rocm.docs.amd.com/projects/rocPRIM/en/latest/intrinsics.html#synchronization

https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#memory-fence-instructions

## Definitions

**Wavefront** 
on nvidia 32 lanes
on amd 64 lanes

**Block** 
Same as a openCL workgroup contains multiple Wavefronts.

## Errors

If you encounter the error 
```
 error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device librar
```
set the `HIP_PLATFORM` to nvidia or amd
```
export HIP_PLATFORM=nvidia
export HIP_PLATFORM=amd
```

If you encounter the error
```
nvcc fatal   : Unknown option '-Wl,-rpath,/usr/local/cuda-12.2/targets/x86_64-linux/lib'
```
unset the `HIP_PLATFORM` environment variable
```
unset HIP_PLATFORM
```