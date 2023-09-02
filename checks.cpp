#include "hip/hip_runtime.h"
#include <iostream>
#include <string>

#define HIP_CHECK(command) {                                                                                  \
    hipError_t status = command;                                                                              \
    if (status != hipSuccess) {                                                                               \
        std::cerr << "Error at " << __LINE__ << ": HIP reports " << hipGetErrorString(status) << std::endl;   \
        std::abort();                                                                                         \
    }                                                                                                         \
}

__global__ void latency() {
    ;
}

__global__ void timeout(int* a) {
    for (size_t i = 0; i < 100000; i++) {
        atomicAdd(a, 1);
    }
}

int main(int argc, char** argv) {
    dim3 work_group_count = dim3(4, 1, 1);
    dim3 work_group_size = dim3(256, 1, 1);
    if (argc == 2 && strcmp(argv[1], "timeout") == 0) {
        hipStream_t stream_1;
        HIP_CHECK(hipStreamCreate(&stream_1));
        hipEvent_t start_1;
        hipEvent_t end_1;
        HIP_CHECK(hipEventCreate(&start_1));
        HIP_CHECK(hipEventCreate(&end_1));
        int *device_data;
        HIP_CHECK(hipMalloc(&device_data, sizeof(int)));
        HIP_CHECK(hipEventRecord(start_1, stream_1));
        hipLaunchKernelGGL(timeout, work_group_count, work_group_size, 0, stream_1, device_data);
        HIP_CHECK(hipEventRecord(end_1, stream_1));
        HIP_CHECK(hipStreamSynchronize(stream_1));
        float time_1;
        HIP_CHECK(hipEventElapsedTime(&time_1, start_1, end_1));
        std::cout << "latency: " << time_1 << "ms" << std::endl;
    } else {
        std::cout << "The timeout test is disabled by default and should only be enabled when the GPU is not running a desktop environment" << std::endl;
        std::cout << "To enable the timeout test use the flag \"timeout\"." << std::endl;
    }
    // int *end = 0;
    // hipStream_t stream_1;
    // hipStream_t stream_2;
    // HIP_CHECK(hipStreamCreate(&stream_1));
    // HIP_CHECK(hipStreamCreate(&stream_2));
    // int *device_end;
    // HIP_CHECK(hipMalloc(&device_end, sizeof(int)));
    // hipLaunchKernelGGL(timeout, work_group_count, work_group_size, 0, stream_1, end);
    // HIP_CHECK(hipGetLastError());
    // *end = 1;
    // HIP_CHECK(hipMemcpyAsync(device_end, end, sizeof(int), hipMemcpyHostToDevice, stream_2));  
    // HIP_CHECK(hipStreamSynchronize(stream_1));
    // HIP_CHECK(hipStreamSynchronize(stream_2));

    hipEvent_t start;
    hipEvent_t end;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventRecord(start, stream));
    hipLaunchKernelGGL(latency, work_group_count, work_group_size, 0, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    float time;
    HIP_CHECK(hipEventElapsedTime(&time, start, end));
    std::cout << "latency: " << time << "ms" << std::endl;
}