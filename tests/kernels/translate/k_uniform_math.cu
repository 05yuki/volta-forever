// Kernels whose sm_75 code computes on uniform values (the same in every thread: parameters, loop
// counters) in as many ways as could be provoked: conversions, bit counts, shifts, 64-bit sums and
// products, comparisons into uniform predicates, selections and branches on them. For the
// differential test of the translator (diffrun.c); integers only in what is compared, so that two
// compilers must agree.
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_uniform_math.sm70.cubin k_uniform_math.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_uniform_math.sm75.cubin k_uniform_math.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

// Integer arithmetic on loop counters and parameters: products, shifts, bit counts, selections.
extern "C" __global__ void k_uniform_int(int32_t* out, const int32_t* x, int n, int rounds, uint32_t seed, int64_t step) {
    int i = I;
    if (i >= n) return;
    int32_t acc = x[i];
    uint64_t wide = seed;
    for (int r = 0; r < rounds; r++) {
        uint32_t u = seed * (r + 1) + (seed >> (r & 15));           // uniform
        int bits = __popc(u) + __clz(u | 1) + __ffs(u);                // uniform bit counts
        wide = wide * 6364136223846793005ull + step + r;               // uniform 64-bit product and sum
        uint32_t m = (uint32_t)(wide >> 33);
        int32_t lo = (int32_t)(m % 1000u) - 500, hi = lo + (r % 7) * 50; // uniform division by a constant
        if ((u & 3) == 1) acc ^= bits;                                  // a branch on a uniform condition
        acc = acc < lo ? lo + (acc & 7) : (acc > hi ? hi - (acc & 3) : acc); // clamp to uniform bounds
        acc = acc * (int32_t)(m | 1) + (int32_t)(wide >> (r & 31));
        acc += (r & 1) ? bits << (r & 7) : -(bits >> 1);
    }
    out[i] = acc;
}

// Floating point with uniform operands; the result is compared as integers after a rounding that
// two compilers cannot differ on (sums and products of small integers, exact in float).
extern "C" __global__ void k_uniform_float(int32_t* out, const int32_t* x, int n, int rounds, float scale, double bias) {
    int i = I;
    if (i >= n) return;
    float acc = (float)(x[i] & 1023);
    double wide = (double)(x[i] & 255);
    for (int r = 0; r < rounds; r++) {
        float fr = (float)r;                       // conversion of a uniform integer
        float s = scale * fr + 3.0f;               // uniform float arithmetic
        acc = acc + s * 2.0f - fr;                 // per-thread with uniform operands
        if (acc > 4096.0f) acc -= 4096.0f + fr;    // comparison with a uniform value
        wide = wide * 0.5 + bias + (double)r;      // doubles, with a uniform sum
        if (wide > 1.0e6) wide -= 1.0e6;
    }
    out[i] = (int32_t)acc * 31 + (int32_t)(wide * 16.0);
}
