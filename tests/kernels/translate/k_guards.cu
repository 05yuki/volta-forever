// Kernels whose sm_75 code has instructions on uniform registers under a guard: a value that is
// the same in every thread, changed under a condition that is the same in every thread (the
// compiler makes `@UP0 UIADD3 UR6, UR4, URZ, URZ` of it). What such an instruction writes is
// written or left as it was, and what stands for the uniform register must follow either way,
// wherever the translation keeps it. For the differential test of the translator (diffrun.c).
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_guards.sm70.cubin k_guards.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_guards.sm75.cubin k_guards.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

// A count that stops at a bound: both are the same in every thread.
extern "C" __global__ void k_guarded_count(int32_t* out, const int32_t* x, int n, int rounds, int limit) {
    int i = I;
    if (i >= n) return;
    int count = 0;
    for (int r = 0; r < rounds; r++) {
        if (count < limit)
            count += 1 + (r & 1);
    }
    out[i] = x[i] + count;
}

// The same on 64 bits, by a step that is a parameter.
extern "C" __global__ void k_guarded_count64(int64_t* out, const int32_t* x, int n, int rounds, int64_t limit, int64_t step) {
    int i = I;
    if (i >= n) return;
    int64_t count = 0;
    for (int r = 0; r < rounds; r++) {
        if (count < limit)
            count += step + r;
    }
    out[i] = (int64_t)x[i] + count;
}

// A state that steps one way or the other by its own lowest bit.
extern "C" __global__ void k_guarded_state(uint32_t* out, const int32_t* x, int n, int rounds, uint32_t seed, uint32_t mask) {
    int i = I;
    if (i >= n) return;
    uint32_t s = seed;
    for (int r = 0; r < rounds; r++) {
        if (s & 1)
            s = (s >> 1) ^ mask;
        else
            s = (s >> 1) + r;
    }
    out[i] = (uint32_t)x[i] ^ s;
}
