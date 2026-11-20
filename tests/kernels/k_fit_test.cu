// Kernels for the test of the second translation at launch (tests/kernels/fitrun.c):
// compiled for sm_75 they declare 64 registers or fewer, so a block of 1,024
// threads can be launched; translated for sm_70 the usual way they take more (their uniform
// registers get registers above the function's own), and the driver refuses that block
// (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES); translated within the budget they fit again.
//
// What gives a kernel uniform registers: loops whose bounds and index arithmetic come from the
// parameters alone (the same in every thread), as in a pooling kernel. What gives it many
// registers: NACC values of the thread alive through those loops.
//
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -DNACC=40 -cubin -o k_fit_test.135.sm75.cubin k_fit_test.cu
#ifndef NACC
#define NACC 40
#endif

struct Shape {
    int kd, kh, kw;      // the loops' bounds
    int sd, sh, sw;      // strides
    int pd, ph, pw;      // offsets
    int dd, dh, dw;      // what one step is worth in each direction
};

__device__ __forceinline__ float walk(const float *in, int tid, int width, Shape s) {
    float a[NACC];
    #pragma unroll
    for (int k = 0; k < NACC; k++)
        a[k] = in[(tid + k) % width];
    for (int d = 0; d < s.kd; d++)
        for (int h = 0; h < s.kh; h++)
            for (int w = 0; w < s.kw; w++) {
                // The same in every thread:
                int at = (d * s.sd + s.pd) * s.dd + (h * s.sh + s.ph) * s.dh + (w * s.sw + s.pw) * s.dw;
                float x = in[(tid + at) % width];
                #pragma unroll
                for (int k = 0; k < NACC; k++)
                    a[k] = a[k] * 0.5f + x * (float)(k + 1);
            }
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < NACC; k++)
        sum += a[k];
    return sum;
}

extern "C" __global__ void
k_fit_regs(float *out, const float *in, int n, int width, Shape s) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = walk(in, tid, width, s);
}

// The same, with the block's dynamic shared memory used at its far end: thread t writes its sum
// in the last `blockDim.x` words of `words`, and adds what its neighbour wrote. With `words` at
// 16384 (64 kB) the launch needs CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES raised on the
// kernel first: a second translation that the layer launches instead must have been given the
// attribute too.
extern "C" __global__ void
k_fit_shared(float *out, const float *in, int n, int width, Shape s, int words) {
    extern __shared__ float buf[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = tid < n ? walk(in, tid, width, s) : 0.0f;
    buf[words - 1 - threadIdx.x] = sum;
    __syncthreads();
    float other = buf[words - 1 - (threadIdx.x + 1) % blockDim.x];
    if (tid < n) out[tid] = sum + other;
}

// Data of the module that kernels write and read: an initialised variable and one that is not.
// A second translation is a second module, with data of its own: the layer must NOT launch
// from it a kernel that names such data (it would read what nobody wrote). `k_fit_reads_global`
// in blocks of 1,024 threads must stay refused.
__device__ float fit_scale = 1.0f;
__device__ float fit_seen[4];

extern "C" __global__ void
k_fit_writes_global(float scale, float seen) {
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        if (threadIdx.x == 0) fit_scale = scale;
        fit_seen[threadIdx.x] = seen + (float)threadIdx.x;
    }
}

extern "C" __global__ void
k_fit_reads_global(float *out, const float *in, int n, int width, Shape s) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = walk(in, tid, width, s) * fit_scale + fit_seen[tid & 3];
}

// A kernel that needs nothing of this.
extern "C" __global__ void
k_fit_simple(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = in[tid] * 2.0f;
}
