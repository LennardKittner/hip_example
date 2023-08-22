#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>

namespace fs = std::filesystem;

#define HIP_CHECK(command) {                                                                                  \
    hipError_t status = command;                                                                              \
    if (status != hipSuccess) {                                                                               \
        std::cerr << "Error at " << __LINE__ << ": HIP reports " << hipGetErrorString(status) << std::endl;   \
        std::abort();                                                                                         \
    }                                                                                                         \
}

__global__ void mem_set(int lanes, int *to) {
    int global = threadIdx.x + blockIdx.x * blockDim.x;
    if (global < lanes) {
        printf("lane: %d content: %d \n", global, to[global]);
        to[global] = global;
    }
}

int main(int argc, char **argv) {
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
    void *to_mapping = mmap(nullptr, to_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_to, 0);
    if (to_mapping == MAP_FAILED) {
        std::cerr << "Failed to mmap: " << strerror(errno) << std::endl;
        close(fd_to);
        return 1;
    }

    HIP_CHECK(hipHostRegister(to_mapping, to_size, 0)); //0x02
    HIP_CHECK(hipHostGetDevicePointer(&gpu_mapping, to_mapping, 0));

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
    return 0;
}