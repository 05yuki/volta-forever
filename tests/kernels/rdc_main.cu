#include <cuda_runtime.h>
#include <cstdio>

extern __device__ int add_from_other_object(int);
extern __device__ void cluster_in_other_object();

extern "C" __global__ void set_five(int *value) {
    *value = 5;
}

__global__ void tally(int *value) {
    *value = add_from_other_object(*value);
}

extern "C" __global__ void needs_cluster() {
    cluster_in_other_object();
}

extern "C" __global__ void via_argument(void (*callee)()) {
    callee();
}

int main() {
    int value = 0;
    int *device = nullptr;
    if (cudaMalloc(&device, sizeof value) != cudaSuccess ||
        cudaMemcpy(device, &value, sizeof value, cudaMemcpyHostToDevice) != cudaSuccess)
        return 1;
    set_five<<<1, 1>>>(device);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(&value, device, sizeof value, cudaMemcpyDeviceToHost) != cudaSuccess || value != 5)
        return 1;
    tally<<<1, 1>>>(device);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(&value, device, sizeof value, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 1;
    std::printf("kernel using another CUDA object: %d\n", value);
    if (cudaFree(device) != cudaSuccess || value != 42)
        return 1;

    // Query only. A lookup that succeeds would give the program a kernel that traps.
    cudaFuncAttributes attributes{};
    cudaError_t status = cudaFuncGetAttributes(&attributes, needs_cluster);
    if (status != cudaErrorInvalidPtx) {
        std::printf("cross-object lookup returned %s\n", cudaGetErrorName(status));
        return 1;
    }
    std::puts("cross-object cluster kernel refused");
    status = cudaFuncGetAttributes(&attributes, via_argument);
    if (status != cudaErrorInvalidPtx) {
        std::printf("indirect lookup returned %s\n", cudaGetErrorName(status));
        return 1;
    }
    std::puts("unresolved indirect kernel refused");
    return 0;
}
