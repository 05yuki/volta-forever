// Test kernels for the sm_75->sm_70 translator, covering every rewriting rule.
// Each kernel is compiled for both sm_70 (reference) and sm_75 (to be translated).
// The translated sm_75 cubin must produce bit-identical output to the sm_70 cubin
// when loaded directly on a V100 with the driver API.
//
// Compile:
//   nvcc-12.9.2 -arch=sm_70 -cubin -fmad=false -o k_translate.sm70.cubin k_translate.cu
//   nvcc-13.5.0 -arch=sm_75 -cubin -fmad=false -o k_translate.sm75.cubin k_translate.cu
//
// These kernels cover:
//   1. IMNMX -> ISETP + SEL  (integer min/max, signed and unsigned)
//   2. IMAD.HI -> IMAD.WIDE + MOV  (high-part multiplication, signed and unsigned)
//   3. S2UR -> S2R  (special registers through uniform path)
//   4. Uniform registers and predicates  (warp-uniform computations)
//   5. Scaled addresses  (shared memory with X4 scaling)
//   6. Branches around rewritten instructions  (forward and backward)
//   7. Loops with rewritten instructions inside
//   8. Shared memory, atomics, __syncthreads
//   9. Warp shuffles
//  10. FP16 arithmetic (non-tensor-core)
//  11. 64-bit arithmetic
//  12. Combined: multiple rewrites in one kernel

#include <cuda_fp16.h>
#include <stdint.h>

// --- 1. IMNMX: integer min/max ---

extern "C" __global__ void k_min_s32(int32_t* out, const int32_t* a, const int32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = min(a[i], b[i]);
}

extern "C" __global__ void k_max_s32(int32_t* out, const int32_t* a, const int32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = max(a[i], b[i]);
}

extern "C" __global__ void k_min_u32(uint32_t* out, const uint32_t* a, const uint32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = min(a[i], b[i]);
}

extern "C" __global__ void k_max_u32(uint32_t* out, const uint32_t* a, const uint32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = max(a[i], b[i]);
}

extern "C" __global__ void k_clamp_s32(int32_t* out, const int32_t* x, int lo, int hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = max(lo, min(x[i], hi));  // two IMNMX in sequence
}

// --- 2. IMAD.HI: high-part multiplication ---

extern "C" __global__ void k_mulhi_s32(int32_t* out, const int32_t* a, const int32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __mulhi(a[i], b[i]);
}

extern "C" __global__ void k_mulhi_u32(uint32_t* out, const uint32_t* a, const uint32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __umulhi(a[i], b[i]);
}

// Division by a constant often uses IMAD.HI for the reciprocal multiply
extern "C" __global__ void k_div_const(uint32_t* out, const uint32_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] / 7u;
}

// --- 3 & 4. Uniform registers / S2UR ---
// These kernels use warp-uniform values (blockIdx, blockDim, kernel params)
// which the compiler places in uniform registers on sm_75.

extern "C" __global__ void k_uniform_addr(float* out, const float* x, int n, int stride) {
    // stride is warp-uniform -> compiler uses UR for the address computation
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int base = blockIdx.x * stride;
        out[i] = x[base + threadIdx.x];
    }
}

extern "C" __global__ void k_uniform_branch(int32_t* out, const int32_t* x, int n, int threshold) {
    // threshold is warp-uniform -> compiler uses UP for the branch predicate
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (blockIdx.x > (unsigned)threshold) {
            out[i] = x[i] * 3 + 1;
        } else {
            out[i] = x[i] * 2;
        }
    }
}

// --- 5. Scaled addresses ---
// Shared-memory accesses with array indexing may use scaled addressing on sm_75.

extern "C" __global__ void k_shared_scale(int32_t* out, const int32_t* x, int n) {
    __shared__ int32_t s[256];  // 4-byte elements -> [R.X4] addressing
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? x[i] : 0;
    __syncthreads();
    // Read from a different index in shared memory
    int j = (t + 1) % blockDim.x;
    int32_t v = s[j];
    __syncthreads();
    s[t] = v * 3 + s[(t + 2) % blockDim.x];
    __syncthreads();
    if (i < n) out[i] = s[t];
}

// --- 6. Branches around rewritten instructions ---

extern "C" __global__ void k_branch_minmax(int32_t* out, const int32_t* a, const int32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int32_t va = a[i], vb = b[i];
    int32_t result;

    // Forward branch: skip the min if va > 100
    if (va > 100) {
        result = max(va, vb);  // IMNMX
    } else {
        result = min(va, vb);  // IMNMX
    }

    // Another branch
    if (result < 0) {
        result = -result;
    }
    out[i] = result;
}

// --- 7. Loops with rewritten instructions ---

extern "C" __global__ void k_loop_minmax(int32_t* out, const int32_t* data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int32_t running_min = data[i];
    int32_t running_max = data[i];

    // Backward branches: the loop condition jumps back
    for (int iter = 1; iter < iters && (i + iter * n) < n * iters; iter++) {
        int idx = (i + iter * 137) % n;  // stride to avoid trivial pattern
        int32_t v = data[idx];
        running_min = min(running_min, v);  // IMNMX inside loop
        running_max = max(running_max, v);  // IMNMX inside loop
    }
    out[i] = running_max - running_min;
}

extern "C" __global__ void k_loop_mulhi(uint32_t* out, const uint32_t* x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t acc = x[i];
    for (int j = 0; j < iters; j++) {
        acc = __umulhi(acc, 0xDEADBEEFu) ^ (acc << 3);  // IMAD.HI in loop
    }
    out[i] = acc;
}

// --- 8. Shared memory + atomics + __syncthreads ---

extern "C" __global__ void k_shmem_reduce(int32_t* out, const int32_t* x, int n) {
    __shared__ int32_t s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? x[i] : 0;
    __syncthreads();

    // Reduction with min/max (IMNMX)
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (t < (unsigned)w) {
            s[t] = min(s[t], s[t + w]);  // IMNMX
        }
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

extern "C" __global__ void k_atomic_minmax(int32_t* gmin, int32_t* gmax,
                                            const int32_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicMin(gmin, x[i]);
        atomicMax(gmax, x[i]);
    }
}

// --- 9. Warp shuffles ---

extern "C" __global__ void k_warp_reduce_minmax(int32_t* out_min, int32_t* out_max,
                                                  const int32_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t vmin = (i < n) ? x[i] : 0x7FFFFFFF;
    int32_t vmax = (i < n) ? x[i] : (int32_t)0x80000000;

    // Warp-level min/max reduction with shuffles
    for (int o = 16; o > 0; o >>= 1) {
        int32_t other_min = __shfl_xor_sync(0xFFFFFFFF, vmin, o);
        int32_t other_max = __shfl_xor_sync(0xFFFFFFFF, vmax, o);
        vmin = min(vmin, other_min);  // IMNMX
        vmax = max(vmax, other_max);  // IMNMX
    }
    // Lane 0 writes
    if ((threadIdx.x & 31) == 0) {
        atomicMin(out_min, vmin);
        atomicMax(out_max, vmax);
    }
}

// --- 10. FP16 arithmetic (non-tensor-core) ---

extern "C" __global__ void k_half_arith(uint16_t* out, const uint16_t* a,
                                         const uint16_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half ha = *(const __half*)&a[i];
        __half hb = *(const __half*)&b[i];
        __half result = __hadd(__hmul(ha, hb), __hsub(ha, hb));
        out[i] = *(uint16_t*)&result;
    }
}

// --- 11. 64-bit arithmetic ---

extern "C" __global__ void k_int64_arith(int64_t* out, const int64_t* a,
                                          const int64_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i] + (a[i] ^ b[i]);
    }
}

// --- 12. Combined: multiple rewrites in one kernel ---

extern "C" __global__ void k_combined(int32_t* out, const int32_t* x, int n) {
    __shared__ int32_t s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    // Load to shared (uniform address computation on sm_75)
    s[t] = (i < n) ? x[i] : 0;
    __syncthreads();

    // Clamp with IMNMX
    int32_t v = max(-1000, min(s[t], 1000));

    // High-part multiply
    v = __mulhi(v, 0x12345678);

    // Warp shuffle
    int32_t other = __shfl_xor_sync(0xFFFFFFFF, v, 1);
    v = min(v, other);

    // Write back
    s[t] = v;
    __syncthreads();
    if (i < n) out[i] = s[t];
}
