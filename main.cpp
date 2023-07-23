#include "hip/hip_runtime.h"
#include <iostream>

#define HIP_CHECK(command) {                                                            \
    hipError_t status = command;                                                        \
    if (status != hipSuccess) {                                                         \
        std::cerr << "Error: HIP reports" << hipGetErrorString(status) << std::endl;    \
        std::abort();                                                                   \
    }                                                                                   \
}                                                                                       \

__global__ void my_kernel(int N, double *d_a) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        d_a[i] *= 2;
    }
}

int main() {
    int N = 1000;
    size_t buffer_size = N*sizeof(double);
    double *h_a = (double*) malloc(buffer_size);
    double *d_a = nullptr;
    HIP_CHECK(hipMalloc(&d_a, buffer_size));
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }
    HIP_CHECK(hipMemcpy(d_a, h_a, buffer_size, hipMemcpyHostToDevice));
    dim3 work_group_count = dim3((N+256-1)/256, 1, 1);
    dim3 work_group_size = dim3(256, 1, 1);
    hipLaunchKernelGGL(my_kernel, work_group_count, work_group_size, 0, 0, N, d_a);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_a, d_a, buffer_size, hipMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << std::endl;
    }

    free(h_a);
    HIP_CHECK(hipFree(d_a));
}