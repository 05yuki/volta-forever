// Kernels whose parameters of 8 and 16 bits are read into uniform registers by sm_75 code (ULDC.U8,
// ULDC.S8, ULDC.U16, ULDC.S16), at every place of a word, for the differential test of the
// translator (diffrun.c).
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_params.sm70.cubin k_params.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_params.sm75.cubin k_params.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

extern "C" __global__ void k_small_params(int32_t* out, const int32_t* x, int n,
                                          uint8_t a, int8_t b, uint8_t c, int8_t d,
                                          uint16_t e, int16_t f, int8_t g, uint16_t h, int16_t k, int rounds) {
    int i = I;
    if (i >= n) return;
    int32_t acc = x[i];
    for (int r = 0; r < rounds; r++) {
        acc = acc * (a + 1) + b;
        acc ^= (c << 8) + d;
        acc += e * 3 - f;
        acc = acc * (g | 1) + h - k;
    }
    out[i] = acc;
}

// A table of constants read at a place chosen by a loop counter: ULDC with a uniform index.
extern "C" __global__ void k_const_table(int32_t* out, const int32_t* x, int n, int rounds) {
    static const int32_t table[16] = {3, -7, 11, 13, -17, 19, 23, -29, 31, 37, -41, 43, 47, -53, 59, 61};
    static const int64_t wide[4] = {0x100000001ll, -0x7fffffffll, 0x123456789abll, -5};
    int i = I;
    if (i >= n) return;
    int64_t acc = x[i];
    for (int r = 0; r < rounds; r++) acc = acc * table[r & 15] + table[(r * 5 + 3) & 15] + wide[r & 3];
    out[i] = (int32_t)(acc ^ (acc >> 32));
}

// Arrays given by value, read at a place chosen by a loop counter: ULDC with a uniform index into
// the parameters.
struct Shape { int32_t dims[8]; int64_t strides[8]; };
extern "C" __global__ void k_param_table(int32_t* out, const int32_t* x, int n, Shape s, int rounds) {
    int i = I;
    if (i >= n) return;
    int64_t acc = x[i];
    // `rounds` is at most 8: the arrays are walked in order, as the dimensions of a tensor are.
    for (int d = 0; d < rounds; d++) {
        int64_t q = acc / s.dims[d];
        acc = (acc - q * s.dims[d]) * s.strides[d] + q;
    }
    out[i] = (int32_t)(acc ^ (acc >> 32));
}
