// Test kernels for call translation: __noinline__ device functions, chains of calls,
// and math-library calls that compile to CALL.ABS.NOINC on sm_75.
//
// The compiler must generate actual calls (not inline the functions), which is why
// we use __noinline__. On sm_75, these generate CALL.REL.NOINC or CALL.ABS.NOINC
// depending on whether the callee is in the same .text section or a different one.

#include <stdint.h>
#include <math.h>

// __noinline__ forces the compiler to emit a real call instruction
__device__ __noinline__ int32_t dev_clamp(int32_t x, int32_t lo, int32_t hi) {
    return max(lo, min(x, hi));  // Two IMNMX in the callee
}

__device__ __noinline__ uint32_t dev_mulhi_chain(uint32_t x, uint32_t y) {
    uint32_t hi = __umulhi(x, y);    // IMAD.HI in the callee
    return hi ^ (x + y);
}

__device__ __noinline__ int32_t dev_abs_diff(int32_t a, int32_t b) {
    return max(a, b) - min(a, b);  // Two IMNMX
}

// Chain: calls another __noinline__ function
__device__ __noinline__ int32_t dev_nested(int32_t x, int32_t y, int32_t z) {
    int32_t clamped = dev_clamp(x, y, z);
    return dev_abs_diff(clamped, x);
}

// --- Kernels that call the device functions ---

extern "C" __global__ void k_call_clamp(int32_t* out, const int32_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = dev_clamp(x[i], -100, 100);
    }
}

extern "C" __global__ void k_call_mulhi(uint32_t* out, const uint32_t* a,
                                         const uint32_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = dev_mulhi_chain(a[i], b[i]);
    }
}

extern "C" __global__ void k_call_nested(int32_t* out, const int32_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = dev_nested(x[i], -50, 50);
    }
}

// Math library calls: sin, cos, exp compile to CALL.ABS.NOINC into libdevice sections
extern "C" __global__ void k_call_math(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // These math functions call into separate .text sections via CALL.ABS.NOINC
        out[i] = sinf(v) + cosf(v) + expf(v);
    }
}

// Multiple calls in a loop (the UR registers must survive across calls)
extern "C" __global__ void k_call_loop(int32_t* out, const int32_t* x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int32_t v = x[i];
    for (int j = 0; j < iters; j++) {
        v = dev_clamp(v + j, -1000, 1000);
        v = (int32_t)dev_mulhi_chain((uint32_t)(v + 0x10000), 0xABCDEF01u);
    }
    out[i] = v;
}
