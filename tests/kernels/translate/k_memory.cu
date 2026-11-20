// Memory instructions that sm_75 code has in forms sm_70 lacks, beyond those of k_uniform.cu: atomic
// operations in shared memory at an index (a stride), atomic operations at an address that is a
// parameter (a uniform address), and loads and stores through a pointer that is shared or global
// memory by a parameter (the generic LD and ST, at a uniform address). For the differential test of
// the translator (diffrun.c).
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_memory.sm70.cubin k_memory.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_memory.sm75.cubin k_memory.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

// Counts, maxima and 64-bit sums by block, in shared memory, at places the data choose: ATOMS.ADD
// and ATOMS.MAX at [R.X4 + n], ATOMS.CAST at [R.X8 + n]. Blocks of 256 threads. Each table has one
// operation, so that the result does not depend on the order of the threads.
extern "C" __global__ void k_shared_atomic(int32_t* out, const uint8_t* x, int n) {
    __shared__ int32_t bins[256];
    __shared__ int32_t tops[256];
    __shared__ int64_t wide[256];
    int i = I;
    bins[threadIdx.x & 255] = 0;
    tops[threadIdx.x & 255] = 0;
    wide[threadIdx.x & 255] = 0;
    __syncthreads();
    if (i < n) {
        atomicAdd(&bins[x[i]], 3);
        atomicMax(&tops[x[i]], i & 15);
        atomicAdd((unsigned long long*)&wide[x[i]], 5ull);
    }
    __syncthreads();
    if (i < n) out[i] = bins[threadIdx.x & 255] + tops[threadIdx.x & 255] * 1000 + (int32_t)wide[threadIdx.x & 255];
}

// A sum in out[0], whose old value each thread asks for: ATOMG.ADD at a uniform address. The sum is
// the same in whatever order the threads add.
extern "C" __global__ void k_global_atomic(int32_t* out, const int32_t* x, int n) {
    int i = I;
    if (i >= n || i == 0) return;
    int old = atomicAdd(out, x[i] & 0xff);
    out[i] = old == 0x7fffffff ? 1 : x[i] * 3;
}

// The same with 64 bits in out[0], and a maximum in out[1] (of numbers that are not negative: the
// old value a thread gets is never negative, whatever the order).
extern "C" __global__ void k_global_atomic64(int64_t* out, const int32_t* x, int n) {
    int i = I;
    if (i >= n || i < 2) return;
    unsigned long long old = atomicAdd((unsigned long long*)out, (unsigned long long)(x[i] & 0xfff));
    int m = atomicMax((int*)(out + 1), x[i] & 0xffff);
    out[i] = (old == ~0ull ? 1 : 0) + (m < 0 ? 2 : 0) + x[i] * 5ll;
}

// A pointer that is shared memory (`which` not 0) or global memory: ST and LD at a uniform address.
// Every block writes the same two values.
extern "C" __global__ void k_generic(int32_t* out, const int32_t* x, int n, int which) {
    __shared__ int32_t s[8];
    int i = I;
    int32_t* p = which ? s : out;
    if (threadIdx.x == 0) { p[1] = 5; p[2] = 9; }
    __syncthreads();
    if (i >= n || i < 4) return;
    out[i] = x[i] + p[1] * 3 + p[2];
}

// The same at places the data choose: LD and ST at a register plus a uniform address. `perm` is a
// permutation of 0 to 63, so that every block writes the same value at each place.
extern "C" __global__ void k_generic_index(int32_t* out, const int32_t* x, int n, int which, int32_t* other, const int32_t* perm) {
    __shared__ int32_t s[64];
    int i = I;
    int32_t* p = which ? s : other;
    if (threadIdx.x < 64) p[perm[threadIdx.x] & 63] = threadIdx.x * 7 + 1;
    __syncthreads();
    if (i >= n) return;
    out[i] = x[i] + p[x[i] & 63] * 3 + p[(x[i] >> 6) & 63];
}
