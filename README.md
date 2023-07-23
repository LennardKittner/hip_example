# hip_example
A simple hip example programm.

## Building

```
hipcc -o hip_example main.cpp
```

### cmake
```
mkdrid build
cd build
cmake -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
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