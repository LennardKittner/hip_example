#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>

namespace fs = std::filesystem;

#define HIP_CHECK(command) {                                                            \
    hipError_t status = command;                                                        \
    if (status != hipSuccess) {                                                         \
        std::cerr << "Error at " << __LINE__ << ": HIP reports " << hipGetErrorString(status) << std::endl;   \
        std::abort();                                                                   \
    }                                                                                   \
}

__global__ void mem_set(int lanes, int *to) {
    int global = threadIdx.x + blockIdx.x * blockDim.x;
    if (global < lanes) {
        printf("lane: %d content: %d \n", global, to[global]);
        to[global] = global;
    }
}

int main(int argc, char **argv) {
    int managed_memory = 0;
    int device;
    HIP_CHECK(hipGetDevice(&device));

    HIP_CHECK(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, device));

    if (!managed_memory ) {
        std::cerr << "Managed memory access not supported on the device" << device << std::endl;
        return 1;
    }
    if (argc != 2) {
        std::cerr << "No to file provided" << std::endl;
        return 1;
    }

    fs::path to = fs::path {argv[1]};
    if (!fs::exists(to)) {
        std::cerr << "File not found: " << to << std::endl;
        return 1;
    }
    
    int fd_to = open(to.c_str(), O_RDWR);
    if (fd_to == -1) {
        std::cerr << "Failed to open file: " << to << std::endl;
        return 1;
    }

    size_t to_size = fs::file_size(to);
    if (to_size == static_cast<size_t>(-1)) {
        std::cerr << "Failed to get size: " << to << std::endl;
        close(fd_to);
        return 1;
    }

    void *gpu_mapping;
    HIP_CHECK(hipMallocManaged(&gpu_mapping, to_size));

    void *to_mapping = mmap(gpu_mapping, to_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd_to, 0);
    if (to_mapping == MAP_FAILED) {
        std::cerr << "Failed to mmap: " << to << std::endl;
        close(fd_to);
        HIP_CHECK(hipFree(gpu_mapping));
        return 1;
    }

    std::cout << "device: " << gpu_mapping << std::endl;
    std::cout << "host: " << to_mapping << std::endl;

    close(fd_to);

    int *to_data = reinterpret_cast<int*>(gpu_mapping);
    int lanes = 256;
    dim3 work_group_count = dim3(4, 1, 1);
    dim3 work_group_size = dim3(64, 1, 1);
    hipLaunchKernelGGL(mem_set, work_group_count, work_group_size, 0, 0, lanes, to_data);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(gpu_mapping));
    return 0;
}