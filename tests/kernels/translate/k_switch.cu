// Kernels with a switch that the compiler turns into a jump table: a table of code offsets in the
// constants of the function, read by LDC, and a branch through a register (BRX). PyTorch's
// comparison kernels choose their operation that way. For the differential test of the translator
// (diffrun.c): when code moves, the table, the branch and their record in the container move.
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_switch.sm70.cubin k_switch.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_switch.sm75.cubin k_switch.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

static __device__ __forceinline__ int32_t choose(int which, int32_t v, int32_t w) {
    switch (which) {
        case 0: return v + w;
        case 1: return v * 3 - w;
        case 2: return (v ^ 0x55) + (w >> 1);
        case 3: return v < w ? v : w;            // a minimum: a scratch predicate next to the table
        case 4: return v > w ? v : w;
        case 5: return (int32_t)(((int64_t)v * 0x10001) >> 16);
        case 6: return v / 7 + w % 5;
        case 7: return -v;
        default: return 7;
    }
}

// The selector is an argument: the same in every thread (a uniform branch on Turing).
extern "C" __global__ void k_switch_uniform(int32_t* out, const int32_t* x, int n, int which) {
    int i = I;
    if (i >= n) return;
    out[i] = choose(which, x[i], x[(i + 1) % n]);
}

// The selector differs from thread to thread: the threads of a warp part at the branch.
extern "C" __global__ void k_switch_thread(int32_t* out, const int32_t* x, int n) {
    int i = I;
    if (i >= n) return;
    out[i] = choose((x[i] >> 3) & 15, x[i], x[(i + 1) % n]);
}

// Two tables in one function, the second inside a loop.
extern "C" __global__ void k_switch_twice(int32_t* out, const int32_t* x, int n, int which, int rounds) {
    int i = I;
    if (i >= n) return;
    int32_t acc = choose(which, x[i], 3);
    for (int r = 0; r < rounds; r++)
        acc = choose((acc ^ r) & 7, acc, x[(i + r) % n]);
    out[i] = acc;
}
