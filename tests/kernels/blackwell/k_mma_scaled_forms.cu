// k_mma_scaled.cu: block-scaled MMA kernels for testing the emulation against host models.
// Same conventions as k_fp8.cu and k_fp4.cu: in[thread*nin + r], out[thread*nout + r].
// Build: nvcc -arch=sm_120a -O3 -ptx -o k_mma_scaled.ptx k_mma_scaled.cu
//
// The test runner (tcrun) loads the PTX through the proxy on a V100, sets up inputs in
// host memory and checks the results against a host model of the instruction.
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// ---- kind::mxf4, E2M1, UE8M0 scale, scale_vec::2X, m16n8k64 ----
// in: a0 a1 a2 a3 b0 b1 c0 c1 c2 c3 sa sb = 12 registers
// out: d0 d1 d2 d3 = 4 registers
KERNEL(s_mxf4_2x) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0, c1, c2, c3;
    __builtin_memcpy(&c0, &in[GI*12+6], 4);
    __builtin_memcpy(&c1, &in[GI*12+7], 4);
    __builtin_memcpy(&c2, &in[GI*12+8], 4);
    __builtin_memcpy(&c3, &in[GI*12+9], 4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0, d1=c1, d2=c2, d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    __builtin_memcpy(&out[GI*4+0], &d0, 4);
    __builtin_memcpy(&out[GI*4+1], &d1, 4);
    __builtin_memcpy(&out[GI*4+2], &d2, 4);
    __builtin_memcpy(&out[GI*4+3], &d3, 4);
}

// ---- kind::mxf4nvf4, E2M1, UE4M3 scale, scale_vec::4X, m16n8k64 ----
KERNEL(s_nvf4_4x) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0, c1, c2, c3;
    __builtin_memcpy(&c0, &in[GI*12+6], 4);
    __builtin_memcpy(&c1, &in[GI*12+7], 4);
    __builtin_memcpy(&c2, &in[GI*12+8], 4);
    __builtin_memcpy(&c3, &in[GI*12+9], 4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0, d1=c1, d2=c2, d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    __builtin_memcpy(&out[GI*4+0], &d0, 4);
    __builtin_memcpy(&out[GI*4+1], &d1, 4);
    __builtin_memcpy(&out[GI*4+2], &d2, 4);
    __builtin_memcpy(&out[GI*4+3], &d3, 4);
}

// ---- kind::mxf8f6f4, E4M3, UE8M0 scale, scale_vec::1X, m16n8k32 ----
KERNEL(s_mxf8_e4m3_1x) {
    unsigned a0=in[GI*12+0], a1=in[GI*12+1], a2=in[GI*12+2], a3=in[GI*12+3];
    unsigned b0=in[GI*12+4], b1=in[GI*12+5];
    float c0, c1, c2, c3;
    __builtin_memcpy(&c0, &in[GI*12+6], 4);
    __builtin_memcpy(&c1, &in[GI*12+7], 4);
    __builtin_memcpy(&c2, &in[GI*12+8], 4);
    __builtin_memcpy(&c3, &in[GI*12+9], 4);
    unsigned sa=in[GI*12+10], sb=in[GI*12+11];
    float d0=c0, d1=c1, d2=c2, d3=c3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    __builtin_memcpy(&out[GI*4+0], &d0, 4);
    __builtin_memcpy(&out[GI*4+1], &d1, 4);
    __builtin_memcpy(&out[GI*4+2], &d2, 4);
    __builtin_memcpy(&out[GI*4+3], &d3, 4);
}
