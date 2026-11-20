// Kernels that execute the instructions of PTX written for Ampere which the proxy rewrites for Volta
// (inline PTX), for tcrun.c. Same conventions as k_tensor.cu: every thread reads its operand registers
// from `in`, runs the instructions and writes the result registers to `out`; `aux` is common data.
//
// These are the instructions XLA's Triton kernels use once the program is told compute capability 8.0
// (ZML does): the FP16 product mma.m16n8k16, cp.async, cvt to packed halves, and min/max
// that return NaN; and those llama.cpp's code for Ampere adds: the integer products in Ampere's
// shapes, and the products of bf16, tf32 and f64.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_80 -O3 -ptx -o k_ampere.135.sm80.ptx k_ampere.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)
#define SHARED_WORDS 1024

// ---- mma.m16n8k16, operands in registers --------------------------------------------------------------
// in: a0..a3 b0 b1 c0..c3, out: d0..d3 (f32).
KERNEL(a_mma_f32) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]), "r"(x[8]),
                   "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// in: a0..a3 b0 b1 c0 c1 (f16 pairs), out: d0 d1.
KERNEL(a_mma_f16) {
    const unsigned* x = in + GI * 8;
    unsigned d[2];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
    out[GI * 2] = d[0], out[GI * 2 + 1] = d[1];
}

// ---- mma.m16n8k16, operands loaded by ldmatrix --------------------------------------------------------
// What Triton's kernels do. aux: 1024 words copied to shared memory; in: byte offsets of the rows of A
// and of B as for the ldmatrix kernels of k_tensor.cu, then d0..d3; out: d0..d3.
#define FILL_SHARED_SAME                                                               \
    __shared__ unsigned memory[SHARED_WORDS];                                          \
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i];   \
    __syncthreads();
#define ROWS(word) ((const char*)memory + x[word])
#define A_LD_K16(name, b_form)                                                                                     \
    KERNEL(name) {                                                                                                 \
        FILL_SHARED_SAME                                                                                           \
        const unsigned* x = in + GI * 6;                                                                           \
        unsigned a[4], b[2], d[4] = {x[2], x[3], x[4], x[5]};                                                      \
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"                                   \
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(ROWS(0)));                             \
        asm volatile("ldmatrix.sync.aligned.m8n8.x2" b_form ".b16 {%0, %1}, [%2];"                                 \
                     : "=r"(b[0]), "=r"(b[1]) : "l"(ROWS(1)));                                                     \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "      \
                     "{%8, %9}, {%0, %1, %2, %3};"                                                                 \
                     : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])                                              \
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));                          \
        for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];                                                        \
    }
A_LD_K16(a_ld_k16, "")
A_LD_K16(a_ld_k16_trans, ".trans")

// ---- cp.async -----------------------------------------------------------------------------------------
// in: the byte offset in aux of 16 bytes (a multiple of 16), and how many of them the first copy takes.
// out: the eight words of the thread's place in shared memory after: a 16-byte copy of that amount, an
// 8-byte copy, a 4-byte copy of nothing, a 4-byte copy.
KERNEL(a_cp) {
    __shared__ __align__(16) unsigned memory[64 * 8];
    const unsigned* x = in + GI * 2;
    const char* source = (const char*)aux + x[0];
    volatile unsigned* mine = memory + threadIdx.x * 8;
    for (int i = 0; i < 8; i++) mine[i] = 0xdddddddd;
    unsigned long long to = __cvta_generic_to_shared((unsigned*)mine);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" : : "l"(to), "l"(source), "r"(x[1]));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" : : "l"(to + 16), "l"(source));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 0;" : : "l"(to + 24), "l"(source));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "l"(to + 28), "l"(source));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (int i = 0; i < 8; i++) out[GI * 8 + i] = mine[i];
}
// A copy under a predicate, as CUTLASS writes it. in: the byte offset in aux of 16 bytes, and a word
// whose lowest bit says whether the thread copies. out: the four words of the thread's place after.
// A thread that does not copy gives the instruction an address that no load could read.
KERNEL(a_cp_guarded) {
    __shared__ __align__(16) unsigned memory[64 * 4];
    const unsigned* x = in + GI * 2;
    unsigned copies = x[1] & 1;
    const char* source = copies ? (const char*)aux + x[0] : (const char*)16;
    volatile unsigned* mine = memory + threadIdx.x * 4;
    for (int i = 0; i < 4; i++) mine[i] = 0xdddddddd;
    unsigned long long to = __cvta_generic_to_shared((unsigned*)mine);
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %0, 0;\n\t"
                 "@p cp.async.cg.shared.global.L2::128B [%1], [%2], 16;\n\t}"
                 : : "r"(copies), "l"(to), "l"(source));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = mine[i];
}

// ---- conversions and bounds ---------------------------------------------------------------------------
// in: two f32; out: their halves, the first in the upper 16 bits.
KERNEL(a_pack) {
    const unsigned* x = in + GI * 2;
    float a = __uint_as_float(x[0]), b = __uint_as_float(x[1]);
    unsigned d;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}
// in: two f32; out: max.NaN and min.NaN of them.
KERNEL(a_nan) {
    const unsigned* x = in + GI * 2;
    float a = __uint_as_float(x[0]), b = __uint_as_float(x[1]), high, low;
    asm volatile("max.NaN.f32 %0, %1, %2;" : "=f"(high) : "f"(a), "f"(b));
    asm volatile("min.NaN.f32 %0, %1, %2;" : "=f"(low) : "f"(a), "f"(b));
    out[GI * 2] = __float_as_uint(high), out[GI * 2 + 1] = __float_as_uint(low);
}

// ---- instructions of later targets with a plain equivalent ---------------------------------------------
// in: two f32 (NaNs, infinities and zeros among them); out: the first as a bf16 to nearest, the same
// truncated, both to nearest packed (the first in the upper half), and the upper half of the second
// read back as a bf16.
KERNEL(a_bf16) {
    const unsigned* x = in + GI * 2;
    float a = __uint_as_float(x[0]), b = __uint_as_float(x[1]), back;
    unsigned short near, toward, half = (unsigned short)(x[1] >> 16);
    unsigned both;
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(near) : "f"(a));
    asm volatile("cvt.rz.bf16.f32 %0, %1;" : "=h"(toward) : "f"(a));
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(both) : "f"(a), "f"(b));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(back) : "h"(half));
    unsigned* y = out + GI * 4;
    y[0] = near, y[1] = toward, y[2] = both, y[3] = __float_as_uint(back);
}
// in: two s32; out: min.relu and max.relu of them.
KERNEL(a_relu) {
    const unsigned* x = in + GI * 2;
    unsigned low, high;
    asm volatile("min.relu.s32 %0, %1, %2;" : "=r"(low) : "r"(x[0]), "r"(x[1]));
    asm volatile("max.relu.s32 %0, %1, %2;" : "=r"(high) : "r"(x[0]), "r"(x[1]));
    out[GI * 2] = low, out[GI * 2 + 1] = high;
}
// in: a byte offset in aux; out: the 32 bytes there (rounded down to a multiple of 32), read with one
// 256-bit load and written with one 256-bit store.
KERNEL(a_wide) {
    const char* source = (const char*)aux + (in[GI] & ~31u);
    unsigned r[8];
    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "l"(source));
    unsigned long long q[4];
    for (int i = 0; i < 4; i++) q[i] = r[2 * i] | (unsigned long long)r[2 * i + 1] << 32;
    asm volatile("st.global.v4.b64 [%0], {%1, %2, %3, %4};" : : "l"(out + GI * 8), "l"(q[0]), "l"(q[1]), "l"(q[2]), "l"(q[3]));
}
// ---- integer mma in Ampere's shapes ------------------------------------------------------------------
// m16n8k16 s8*s8: in: a0 a1 b c0..c3, out: d0..d3.
KERNEL(a_i8_k16) {
    const unsigned* x = in + GI * 7;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%7, %8, %9, %10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// m16n8k16 u8*s8: in: a0 a1 b c0..c3, out: d0..d3.
KERNEL(a_u8s8_k16) {
    const unsigned* x = in + GI * 7;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%7, %8, %9, %10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// m16n8k32 s8*s8: in: a0..a3 b0 b1 c0..c3, out: d0..d3.
KERNEL(a_i8_k32) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// m16n8k16 s8*s8 with satfinite, written after the layouts as in the manual: in: a0 a1 b c0..c3,
// out: d0..d3.
KERNEL(a_i8_k16_sat) {
    const unsigned* x = in + GI * 7;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%7, %8, %9, %10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// m16n8k32 s8*s8 with satfinite, written after the types as compilers do for this shape (cuDNN has
// it so): in: a0..a3 b0 b1 c0..c3, out: d0..d3.
KERNEL(a_i8_k32_sat) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// m8n8k16 s8*s8 with satfinite: in: a b c0 c1, out: d0 d1.
KERNEL(a_i8_turing_sat) {
    const unsigned* x = in + GI * 4;
    unsigned d[2];
    asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
    out[GI * 2] = d[0], out[GI * 2 + 1] = d[1];
}
// m16n8k16, A from ldmatrix, B as a register: aux -> shared; in: byte offset for A, B register,
// then d0..d3; out: d0..d3.
KERNEL(a_i8_ld_k16) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 6;
    unsigned a[2], d[4] = {x[2], x[3], x[4], x[5]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(a[0]), "=r"(a[1]) : "l"(ROWS(0)));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                 "{%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(x[1]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// m16n8k32, the four registers of A from one ldmatrix, B as two registers, accumulated in place (how
// a quantized matrix product is written for Ampere): aux -> shared; in: byte offset for A, b0 b1,
// then d0..d3; out: d0..d3.
KERNEL(a_i8_ld_k32) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 7;
    unsigned a[4], d[4] = {x[3], x[4], x[5], x[6]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(ROWS(0)));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(x[1]), "r"(x[2]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// ---- products of bf16, tf32 and f64: each thread computes its sums itself ---------------------------
// Two blocks of k: in: a0..a3 b0 b1 c0..c3, out: d0..d3 (f32).
#define SUMS_TWO(name, form)                                                                                         \
    KERNEL(name) {                                                                                                  \
        const unsigned* x = in + GI * 10;                                                                           \
        unsigned d[4];                                                                                              \
        asm volatile("mma.sync.aligned." form " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};" \
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])                                               \
                     : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]),      \
                       "r"(x[8]), "r"(x[9]));                                                                       \
        for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];                                                         \
    }
// One block of k: in: a0 a1 b c0..c3, out: d0..d3 (f32).
#define SUMS_ONE(name, form)                                                                                         \
    KERNEL(name) {                                                                                                  \
        const unsigned* x = in + GI * 7;                                                                            \
        unsigned d[4];                                                                                              \
        asm volatile("mma.sync.aligned." form " {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"                \
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])                                               \
                     : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]));                \
        for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];                                                         \
    }
SUMS_TWO(s_bf16_k16, "m16n8k16.row.col.f32.bf16.bf16.f32")
SUMS_ONE(s_bf16_k8, "m16n8k8.row.col.f32.bf16.bf16.f32")
SUMS_TWO(s_tf32_k8, "m16n8k8.row.col.f32.tf32.tf32.f32")
SUMS_ONE(s_tf32_k4, "m16n8k4.row.col.f32.tf32.tf32.f32")
// in: a, b, c0, c1 (doubles, two words each); out: d0, d1.
KERNEL(s_f64_k4) {
    const double* x = (const double*)(in + GI * 8);
    double d0, d1;
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=d"(d0), "=d"(d1) : "d"(x[0]), "d"(x[1]), "d"(x[2]), "d"(x[3]));
    double* y = (double*)(out + GI * 4);
    y[0] = d0, y[1] = d1;
}
// Two products in a row with the same B, the second on the results of the first, in place.
// in: a0..a3 of the first, a0..a3 of the second, b0 b1, c0..c3; out: d0..d3.
KERNEL(s_bf16_twice) {
    const unsigned* x = in + GI * 14;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[8]), "r"(x[9]), "r"(x[10]), "r"(x[11]), "r"(x[12]),
                   "r"(x[13]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// in: one f32; out: it rounded to tf32.
KERNEL(s_cvt_tf32) {
    unsigned d;
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(d) : "f"(__uint_as_float(in[GI])));
    out[GI] = d;
}

// in: one s32; out: the sum and the largest of the 32 of the warp.
KERNEL(a_redux) {
    unsigned sum, largest;
    asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(sum) : "r"(in[GI]));
    // The member mask in a register, as compilers write it.
    asm volatile("redux.sync.max.s32 %0, %1, %2;" : "=r"(largest) : "r"(in[GI]), "r"(0xffffffffu));
    out[GI * 2] = sum, out[GI * 2 + 1] = largest;
}
