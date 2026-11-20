// Kernels whose sm_75 code reads and writes memory at addresses with a uniform part (the same in
// every thread: a pointer given to the kernel, a row of a table chosen by a loop counter), for the
// differential test of the translator (diffrun.c).
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_uniform.sm70.cubin k_uniform.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_uniform.sm75.cubin k_uniform.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

// Values read through pointers that are the same for every thread.
extern "C" __global__ void k_uniform_scalars(int32_t* out, const int32_t* x, const int32_t* scale, const int64_t* bias, int n) {
    int i = I;
    if (i >= n) return;
    out[i] = (int32_t)(x[i] * *scale + *bias);
}

// A table walked by a loop counter: its address is uniform; the other operand is per thread.
extern "C" __global__ void k_uniform_table(int32_t* out, const int32_t* x, const int32_t* table, int n, int m) {
    int i = I;
    if (i >= n) return;
    int32_t acc = 0;
    for (int j = 0; j < m; j++) acc += x[i] * table[j] + table[m - 1 - j];
    out[i] = acc;
}

// Rows chosen by a loop counter (uniform base), columns by the thread: [register + uniform].
extern "C" __global__ void k_uniform_rows(int64_t* out, const int64_t* x, int n, int rows) {
    int i = I;
    if (i >= n) return;
    int64_t acc = 0;
    for (int r = 0; r < rows; r++) {
        const int64_t* row = x + (int64_t)r * n;
        acc += row[i] * (r + 1);
        out[(int64_t)r * n + i] = acc;
    }
}

// 16-bit and 8-bit elements, an unsigned 32-bit index.
extern "C" __global__ void k_uniform_small(uint16_t* out, const uint16_t* x, const uint8_t* y, unsigned n, unsigned rows) {
    unsigned i = I;
    if (i >= n) return;
    for (unsigned r = 0; r < rows; r++) out[r * n + i] = (uint16_t)(x[r * n + i] + y[r * n + (n - 1 - i)]);
}
