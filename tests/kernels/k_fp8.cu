// Kernels that execute the FP8 instructions the proxy rewrites for Volta (inline PTX), for tcrun.c.
// Same conventions as k_tensor.cu: every thread reads its operand registers from `in`, runs the
// instructions and writes the result registers to `out`; `aux` is common data.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_89 -O3 -ptx -o k_fp8.135.sm89.ptx k_fp8.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)
#define SHARED_WORDS 1024

// ---- cvt.rn.f16x2.e4m3x2: decode two e4m3 bytes to two halves -----------------------------------
// in: one b16 register (low 16 bits of a word); out: one b32 register (f16x2).
KERNEL(f8_decode_e4m3) {
    unsigned x = in[GI] & 0xffff;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}

// ---- cvt.rn.f16x2.e5m2x2: decode two e5m2 bytes to two halves -----------------------------------
KERNEL(f8_decode_e5m2) {
    unsigned x = in[GI] & 0xffff;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.f16x2.e5m2x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}

// ---- cvt.rn.satfinite.e4m3x2.f32: encode two f32 to packed e4m3 pair ----------------------------
// in: two f32 (a, b); out: one word with the packed e4m3 pair in its low 16 bits.
KERNEL(f8_encode_e4m3) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rn.satfinite.e4m3x2.f32 tmp, %1, %2;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// ---- cvt.rn.satfinite.e5m2x2.f32: encode two f32 to packed e5m2 pair ----------------------------
KERNEL(f8_encode_e5m2) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rn.satfinite.e5m2x2.f32 tmp, %1, %2;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// ---- cvt.rn.satfinite.{e4m3x2,e5m2x2}.f16x2: encode two halves ------------------------------------
// in: one register of two halves; out: the e4m3 pair, then the e5m2 pair.
KERNEL(f8_encode_halves) {
    unsigned x = in[GI], d4, d5;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rn.satfinite.e4m3x2.f16x2 tmp, %1;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d4) : "r"(x));
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rn.satfinite.e5m2x2.f16x2 tmp, %1;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d5) : "r"(x));
    out[GI * 2] = d4, out[GI * 2 + 1] = d5;
}

// ---- two products in a row with the same B, the second on the results of the first ----------------
// in: a0..a3 of the first, a0..a3 of the second, b0 b1, c0..c3; out: d0..d3.
KERNEL(f8_mma_twice) {
    const unsigned* x = in + GI * 14;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[8]), "r"(x[9]),
                   "r"(x[10]), "r"(x[11]), "r"(x[12]), "r"(x[13]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// ---- mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ----------------------------------------
// in: a0..a3 b0 b1 c0..c3; out: d0..d3 (f32).
KERNEL(f8_mma_e4m3_e4m3) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                   "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// ---- mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 ----------------------------------------
KERNEL(f8_mma_e5m2_e4m3) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                   "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// ---- mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 ----------------------------------------
KERNEL(f8_mma_e4m3_e5m2) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                   "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// ---- mma.m16n8k32 with operands loaded from shared memory ----------------------------------------
#define FILL_SHARED                                                                \
    __shared__ unsigned memory[SHARED_WORDS];                                     \
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i]; \
    __syncthreads();

KERNEL(f8_mma_ld_e4m3) {
    FILL_SHARED
    const unsigned* x = in + GI * 6;
    // Load A and B from shared memory through ldmatrix
    unsigned a[4], b[2], d[4] = {x[2], x[3], x[4], x[5]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "l"((const char*)memory + x[0]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1])
                 : "l"((const char*)memory + x[1]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
