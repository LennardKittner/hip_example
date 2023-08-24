#include <iostream>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>

namespace fs = std::filesystem;

__global__ void mem_set(int lanes, int *to) {
    int global = threadIdx.x + blockIdx.x * blockDim.x;

    if (global < lanes) {
        printf("lane1: %d\n", global);
        printf("lane: %d content: %d \n", global, to[global]);
        printf("lane2: %d\n", global);
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

    void *to_mapping = mmap(nullptr, to_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_to, 0);
    if (to_mapping == MAP_FAILED) {
        std::cerr << "Failed to mmap: " << strerror(errno) << std::endl;
        close(fd_to);
        return 1;
    }

    
    auto cuda_retval = cudaHostRegister(to_mapping, to_size, cudaHostRegisterIoMemory);
	if (cuda_retval != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString (cuda_retval) + std::string (" (cudaHostRegister)"));

    close(fd_to);

    int *to_data = reinterpret_cast<int*>(to_mapping);
    int lanes = 256;
    dim3 work_group_count = 4;
    dim3 work_group_size = 64;
    mem_set<<<work_group_count, work_group_size>>>(lanes, to_data);
    cudaDeviceSynchronize();
    cudaHostUnregister(to_mapping);
    if (!munmap(to_mapping, to_size)) {
        std::cerr << "Failed to munmap: " << strerror(errno) << std::endl;
    } //  No such file or directory. Why?
    return 0;
}