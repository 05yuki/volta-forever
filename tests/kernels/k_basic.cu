// Small kernels without tensor-core use: arithmetic, shared memory, atomics, warp shuffle, math calls.
#include <cuda_fp16.h>
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
extern "C" __global__ void mathk(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sinf(x[i]) * expf(-x[i] * x[i]) + sqrtf(fabsf(x[i])) + powf(fabsf(x[i]) + 1.0f, 1.5f);
}
extern "C" __global__ void block_sum(const float* x, float* out, int n) {
    __shared__ float s[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    s[t] = i < n ? x[i] : 0.0f;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (t < w) s[t] += s[t + w];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}
extern "C" __global__ void warp_sum(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? x[i] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, v);
}
extern "C" __global__ void histo(const unsigned* x, unsigned* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[x[i] & 63u], 1u);
}
extern "C" __global__ void half_dot(const __half* a, const __half* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__hmul(a[i], b[i])) + (float)__popc(i) + (float)__clz(i + 1);
}
extern "C" __global__ void dbl(const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fma(x[i], 1.000001, 1.0 / (x[i] * x[i] + 1.0)) + (double)(((long long)i * 2654435761LL) % 1000);
}
