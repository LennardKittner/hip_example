# hip_example

A simple hip example programm.

## Manuel Building

```
hipcc -o hip_example main.cpp
```

### Building With Cmake

If you are using a nvidia gpu first set the `GPU_PLATFORM`
```
export GPU_PLATFORM=nvidia
```
Then build with
```
mkdrid build
cd build
cmake ..
```

## Errors

If you get the error 
```
 error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device librar
```
set the `HIP_PLATFORM` to nvidia or amd
```
export HIP_PLATFORM=nvidia
export HIP_PLATFORM=amd
```