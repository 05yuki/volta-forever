// Minimal IMNMX-in-loop test
#include <stdint.h>

extern "C" __global__ void k_simple_min(int32_t* out, const int32_t* a, const int32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t v = a[i];
        // Simple loop with min inside
        for (int j = 0; j < 4; j++) {
            v = min(v, b[(i + j * 137) % n]);
        }
        out[i] = v;
    }
}
