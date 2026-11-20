// Isolated test: k_loop_minmax only (no RED/ATOM instructions)
#include <stdint.h>

extern "C" __global__ void k_loop_minmax(int32_t* out, const int32_t* data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t running_min = data[i];
    int32_t running_max = data[i];
    for (int iter = 1; iter < iters && (i + iter * n) < n * iters; iter++) {
        int idx = (i + iter * 137) % n;
        int32_t v = data[idx];
        running_min = min(running_min, v);
        running_max = max(running_max, v);
    }
    out[i] = running_max - running_min;
}

extern "C" __global__ void k_loop_mulhi(uint32_t* out, const uint32_t* x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t acc = x[i];
    for (int j = 0; j < iters; j++) {
        acc = __umulhi(acc, 0xDEADBEEFu) ^ (acc << 3);
    }
    out[i] = acc;
}
