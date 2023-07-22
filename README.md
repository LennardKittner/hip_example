# hip_example
A simple hip example programm.

## Building

```
mkdrid build
cd build
cmake -D CMAKE_HIP_COMPILER_ROCM_ROOT:PATH=<Path to rocm> ..
```

One example path to rocm is `/opt/rocm`