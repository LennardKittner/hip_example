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

struct memset_cmdlist {
    size_t size;
    size_t num_groups;
    size_t num_lanes_per_group;
    size_t pattern;
    int done;
    int error;

    memset_cmdlist(size_t size, size_t num_groups, size_t num_lanes_per_group, size_t pattern)
        : size(size), num_groups(num_groups), num_lanes_per_group(num_lanes_per_group), pattern(pattern), done(0), error(0) {}
};

__global__ void kernel_memset(memset_cmdlist *args, size_t *to) {
    int global = threadIdx.x + blockIdx.x * blockDim.x;
    size_t elements = args->size / sizeof(size_t); // size is rounded down to a multiple of sizeof(size_t)
    if (elements < args->num_groups * args->num_lanes_per_group) {
        if (global < elements) {
            to[global] = args->pattern;
        }
    } else {
        size_t elements_per_lane = elements / (args->num_groups * args->num_lanes_per_group); // elements is rounded down to a multiple of elements_per_lane
        for (size_t i = 0; i < elements_per_lane; i++) {
            to[global * elements_per_lane + i] = args->pattern;
        }
    }
    // atomicAdd_system(&args->done, 1); // ~20ms for 256 lanes or ~7% for 512MB
}

__global__ void kernel_memset_loop(memset_cmdlist *args, size_t *to, int repeat) {
    for (int i = 0; i < repeat; i++) {
        int global = threadIdx.x + blockIdx.x * blockDim.x;
        size_t elements = args->size / sizeof(size_t); // size is rounded down to a multiple of sizeof(size_t)
        if (elements < args->num_groups * args->num_lanes_per_group) {
            if (global < elements) {
                to[global] = args->pattern;
            }
        } else {
            size_t elements_per_lane = elements / (args->num_groups * args->num_lanes_per_group); // elements is rounded down to a multiple of elements_per_lane
            for (size_t i = 0; i < elements_per_lane; i++) {
                to[global * elements_per_lane + i] = args->pattern;
            }
        }
    }
    // atomicAdd_system(&args->done, 1); // ~20ms for 256 lanes or ~7% for 512MB
}

// only submit the kernel one time and loop on the gpu
void memset_with_timing_loop_on_gpu(memset_cmdlist *args, size_t *to, hipStream_t *stream, int repeats_kernel) {
    // events for timing
    hipEvent_t start;
    hipEvent_t end_cpy_h_d;
    hipEvent_t end_kernel;
    hipEvent_t end_cpy_d_h;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end_cpy_h_d));
    HIP_CHECK(hipEventCreate(&end_kernel));
    HIP_CHECK(hipEventCreate(&end_cpy_d_h));

    dim3 work_group_count = dim3(args->num_groups, 1, 1);
    dim3 work_group_size = dim3(args->num_lanes_per_group, 1, 1);
    // copy data to device
    memset_cmdlist *device_data;
    HIP_CHECK(hipEventRecord(start, *stream));
    HIP_CHECK(hipMalloc(&device_data, sizeof(memset_cmdlist)));
    HIP_CHECK(hipMemcpyAsync(device_data, args, sizeof(memset_cmdlist), hipMemcpyHostToDevice, *stream));
    HIP_CHECK(hipEventRecord(end_cpy_h_d, *stream));
    // start kernel
    hipLaunchKernelGGL(kernel_memset_loop, work_group_count, work_group_size, 0, *stream, device_data,to, repeats_kernel);
    // HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(end_kernel, *stream));
    // copy data from device
    HIP_CHECK(hipMemcpyAsync(args, device_data, sizeof(memset_cmdlist), hipMemcpyDeviceToHost, *stream))
    HIP_CHECK(hipEventRecord(end_cpy_d_h, *stream));
    //TODO: free mem on device
    HIP_CHECK(hipStreamSynchronize(*stream));
    HIP_CHECK(hipFree(device_data));
    // read timings
    float time1;
    float time2;
    float time3;
    HIP_CHECK(hipEventElapsedTime(&time1, start, end_cpy_h_d));
    HIP_CHECK(hipEventElapsedTime(&time2, end_cpy_h_d, end_kernel));
    HIP_CHECK(hipEventElapsedTime(&time3, end_kernel, end_cpy_d_h));
    std::cout << "time copy host device: " << time1 << "ms" << std::endl;
    std::cout << "time memset: " << time2 / repeats_kernel << "ms" << std::endl;
    std::cout << "time copy device host: " << time3 << "ms" << std::endl;
    std::cout << "total time: " << time1 + time2 + time3 << "ms" << std::endl;
    HIP_CHECK(hipEventDestroy(start))
    HIP_CHECK(hipEventDestroy(end_cpy_h_d))
    HIP_CHECK(hipEventDestroy(end_kernel))
    HIP_CHECK(hipEventDestroy(end_cpy_d_h))
}

// submit multiple kernel single stream
void memset_with_timing_multi_kernel(memset_cmdlist *args, size_t *to, hipStream_t *stream, int repeats_kernel) {
    // events for timing
    hipEvent_t start;
    hipEvent_t end_cpy_h_d;
    hipEvent_t end_kernel;
    hipEvent_t end_cpy_d_h;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end_cpy_h_d));
    HIP_CHECK(hipEventCreate(&end_kernel));
    HIP_CHECK(hipEventCreate(&end_cpy_d_h));

    dim3 work_group_count = dim3(args->num_groups, 1, 1);
    dim3 work_group_size = dim3(args->num_lanes_per_group, 1, 1);
    // copy data to device
    memset_cmdlist *device_data;
    HIP_CHECK(hipEventRecord(start, *stream));
    HIP_CHECK(hipMalloc(&device_data, sizeof(memset_cmdlist)));
    HIP_CHECK(hipMemcpyAsync(device_data, args, sizeof(memset_cmdlist), hipMemcpyHostToDevice, *stream));
    HIP_CHECK(hipEventRecord(end_cpy_h_d, *stream));
    // start kernel
    for (int i = 0; i < repeats_kernel; i++) {
        hipLaunchKernelGGL(kernel_memset, work_group_count, work_group_size, 0, *stream, device_data,to);
        // HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(end_kernel, *stream));
    // copy data from device
    HIP_CHECK(hipMemcpyAsync(args, device_data, sizeof(memset_cmdlist), hipMemcpyDeviceToHost, *stream))
    HIP_CHECK(hipEventRecord(end_cpy_d_h, *stream));
    //TODO: free mem on device
    HIP_CHECK(hipStreamSynchronize(*stream));
    HIP_CHECK(hipFree(device_data));
    // read timings
    float time1;
    float time2;
    float time3;
    HIP_CHECK(hipEventElapsedTime(&time1, start, end_cpy_h_d));
    HIP_CHECK(hipEventElapsedTime(&time2, end_cpy_h_d, end_kernel));
    HIP_CHECK(hipEventElapsedTime(&time3, end_kernel, end_cpy_d_h));
    std::cout << "time copy host device: " << time1 << "ms" << std::endl;
    std::cout << "time memset: " << time2 / repeats_kernel << "ms" << std::endl;
    std::cout << "time copy device host: " << time3 << "ms" << std::endl;
    std::cout << "total time: " << time1 + time2 + time3 << "ms" << std::endl;
    HIP_CHECK(hipEventDestroy(start))
    HIP_CHECK(hipEventDestroy(end_cpy_h_d))
    HIP_CHECK(hipEventDestroy(end_kernel))
    HIP_CHECK(hipEventDestroy(end_cpy_d_h))
}

void memset(memset_cmdlist *args, size_t *to, hipStream_t *stream) {
    dim3 work_group_count = dim3(args->num_groups, 1, 1);
    dim3 work_group_size = dim3(args->num_lanes_per_group, 1, 1);

    // copy data to device
    memset_cmdlist *device_data;
    HIP_CHECK(hipMalloc(&device_data, sizeof(memset_cmdlist)));
    HIP_CHECK(hipMemcpyAsync(device_data, args, sizeof(memset_cmdlist), hipMemcpyHostToDevice, *stream));
    // start kernel
    hipLaunchKernelGGL(kernel_memset, work_group_count, work_group_size, 0, *stream, device_data,to);
    HIP_CHECK(hipGetLastError());
    // copy data from device
    HIP_CHECK(hipMemcpyAsync(args, device_data, sizeof(memset_cmdlist), hipMemcpyDeviceToHost, *stream))
    //TODO: free mem on device
    //Maybe include device mem in args and free if args is freed
}

// args: to_file file_size
int main(int argc, char **argv) {
    if (argc < 2) {
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
    if (argc >= 3) {
        to_size = std::strtoul(argv[2], nullptr, 10);
    } else {
        to_size = fs::file_size(to);
        if (to_size == static_cast<size_t>(-1)) {
            std::cerr << "Failed to get size: " << to << std::endl;
            close(fd_to);
            return 1;
        }
    }
    int repeats = 100;
    if (argc >= 4) {
        repeats = std::strtoul(argv[3], nullptr, 10);
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

    // execute memset
    hipStream_t main_stream_1;
    HIP_CHECK(hipStreamCreate(&main_stream_1));
    memset_cmdlist* cmd;
    HIP_CHECK(hipHostMalloc(&cmd, sizeof(memset_cmdlist)));
    *cmd = memset_cmdlist(to_size, 4, 64, 0xabcdef12);
    size_t *to_mapping_gpu = static_cast<size_t*>(gpu_mapping);
    std::cout << "loop kernel on single stream" << std::endl;
    memset_with_timing_multi_kernel(cmd, to_mapping_gpu, &main_stream_1, repeats);
    HIP_CHECK(hipStreamSynchronize(main_stream_1));
    hipStream_t main_stream_2;
    HIP_CHECK(hipStreamCreate(&main_stream_2));
    std::cout << "loop inside kernel" << std::endl;
    memset_with_timing_loop_on_gpu(cmd, to_mapping_gpu, &main_stream_2, repeats);
    HIP_CHECK(hipStreamSynchronize(main_stream_2));

    // clean up
    HIP_CHECK(hipStreamDestroy(main_stream_1));
    HIP_CHECK(hipStreamDestroy(main_stream_2));
    HIP_CHECK(hipHostFree(cmd));
    HIP_CHECK(hipHostUnregister(to_mapping));
    if (!munmap(to_mapping, to_size)) {
        std::cerr << "Failed to munmap: " << strerror(errno) << std::endl;
    } //TODO:  No such file or directory. Why?
    return 0;
}