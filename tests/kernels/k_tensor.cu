// Kernels that execute one tensor-core instruction form each (inline PTX), for tcrun.c.
//
// Every thread reads its operand registers from `in` (NIN words per thread, thread index
// blockIdx.x * blockDim.x + threadIdx.x), runs the instruction and writes the result registers to `out`
// (NOUT words per thread). `aux` is data common to all threads. Registers are passed as raw 32-bit
// words: the harness packs and unpacks f16 pairs, f32 and int8 quadruples.
//
// v884_*: Volta's own mma.m8n8k4 in its four layouts and three accumulator type pairs (compiled for
//         sm_75 as well; on a V100 the kernels check the fragment layouts the rewriting relies on).
// t_*:    the instructions that start with Turing, which the proxy rewrites for Volta.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -ptx -o k_tensor.135.sm75.ptx k_tensor.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// ---- Volta: mma.m8n8k4 ------------------------------------------------------------------------------
// in: a0 a1 b0 b1 c0..c7 (12 words), out: d0..d7.
#define V884_F32F32(name, layout)                                                                                 \
    KERNEL(name) {                                                                                                \
        const unsigned* x = in + GI * 12;                                                                         \
        unsigned d[8];                                                                                            \
        asm volatile("mma.sync.aligned.m8n8k4." layout ".f32.f16.f16.f32 "                                        \
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};" \
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]) \
                     : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]),    \
                       "r"(x[8]), "r"(x[9]), "r"(x[10]), "r"(x[11]));                                             \
        for (int i = 0; i < 8; i++) out[GI * 8 + i] = d[i];                                                       \
    }
// in: a0 a1 b0 b1 c0..c3 (8 words, f16 pairs), out: d0..d3 (f16 pairs).
#define V884_F16F16(name, layout)                                                                                 \
    KERNEL(name) {                                                                                                \
        const unsigned* x = in + GI * 8;                                                                          \
        unsigned d[4];                                                                                            \
        asm volatile("mma.sync.aligned.m8n8k4." layout ".f16.f16.f16.f16 "                                        \
                     "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};"                                  \
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])                                             \
                     : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));   \
        for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];                                                       \
    }
// in: a0 a1 b0 b1 c0..c3 (8 words, f16 pairs), out: d0..d7 (f32).
#define V884_F32F16(name, layout)                                                                                 \
    KERNEL(name) {                                                                                                \
        const unsigned* x = in + GI * 8;                                                                          \
        unsigned d[8];                                                                                            \
        asm volatile("mma.sync.aligned.m8n8k4." layout ".f32.f16.f16.f16 "                                        \
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%12, %13, %14, %15};"              \
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]) \
                     : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));   \
        for (int i = 0; i < 8; i++) out[GI * 8 + i] = d[i];                                                       \
    }
#define V884(layout_name, layout)                 \
    V884_F32F32(v884_##layout_name##_f32f32, layout) \
    V884_F16F16(v884_##layout_name##_f16f16, layout) \
    V884_F32F16(v884_##layout_name##_f32f16, layout)
V884(rowrow, "row.row")
V884(rowcol, "row.col")
V884(colrow, "col.row")
V884(colcol, "col.col")

// ---- Turing: one instruction ------------------------------------------------------------------------
// in: a0 a1 b c0..c3, out: d0..d3 (f32).
KERNEL(t_mma_f32) {
    const unsigned* x = in + GI * 7;
    unsigned d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]));
    unsigned* y = out + GI * 4;
    y[0] = d0, y[1] = d1, y[2] = d2, y[3] = d3;
}
// in: a0 a1 b c0 c1, out: d0 d1 (f16 pairs).
KERNEL(t_mma_f16) {
    const unsigned* x = in + GI * 5;
    unsigned d0, d1;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};"
                 : "=r"(d0), "=r"(d1)
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]));
    out[GI * 2] = d0, out[GI * 2 + 1] = d1;
}
// in: a b c0 c1, out: d0 d1 (s32); elements: int8 x int8, uint8 x int8, int8 x uint8, uint8 x uint8.
#define T_MMA_BYTES(name, types)                                                                         \
    KERNEL(name) {                                                                                       \
        const unsigned* x = in + GI * 4;                                                                 \
        unsigned d0, d1;                                                                                 \
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32." types ".s32 {%0, %1}, {%2}, {%3}, {%4, %5};" \
                     : "=r"(d0), "=r"(d1)                                                                \
                     : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));                                      \
        out[GI * 2] = d0, out[GI * 2 + 1] = d1;                                                          \
    }
T_MMA_BYTES(t_mma_s8s8, "s8.s8")
T_MMA_BYTES(t_mma_u8s8, "u8.s8")
T_MMA_BYTES(t_mma_s8u8, "s8.u8")
T_MMA_BYTES(t_mma_u8u8, "u8.u8")
// in: a, out: d (f16 pair or any two 16-bit elements).
KERNEL(t_movmatrix) {
    unsigned d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(in[GI]));
    out[GI] = d;
}

// ---- Turing: ldmatrix -------------------------------------------------------------------------------
// aux: 1024 words copied to shared memory (each block adds its index to every 16-bit element);
// in: byte offset in shared memory of the matrix row this thread gives the address of; out: 1, 2 or 4
// registers. The addresses are generic, or with ".shared" 32-bit and 64-bit shared-memory offsets.
#define SHARED_WORDS 1024
#define FILL_SHARED                                                                                        \
    __shared__ unsigned memory[SHARED_WORDS];                                                              \
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i] + blockIdx.x * 0x00010001u; \
    __syncthreads();
#define T_LDMATRIX(name, form, count, results, operands)                               \
    KERNEL(name) {                                                                     \
        FILL_SHARED                                                                    \
        const char* p = (const char*)memory + in[GI];                                  \
        unsigned r[4] = {0, 0, 0, 0};                                                  \
        asm volatile("ldmatrix.sync.aligned.m8n8." form ".b16 " results ", [%" #count "];" : operands : "l"(p)); \
        for (int i = 0; i < count; i++) out[GI * count + i] = r[i];                    \
    }
#define R1 "=r"(r[0])
#define R2 "=r"(r[0]), "=r"(r[1])
#define R4 "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
T_LDMATRIX(t_ldmatrix_x1, "x1", 1, "{%0}", R1)
T_LDMATRIX(t_ldmatrix_x2, "x2", 2, "{%0, %1}", R2)
T_LDMATRIX(t_ldmatrix_x4, "x4", 4, "{%0, %1, %2, %3}", R4)
T_LDMATRIX(t_ldmatrix_x1_trans, "x1.trans", 1, "{%0}", R1)
T_LDMATRIX(t_ldmatrix_x2_trans, "x2.trans", 2, "{%0, %1}", R2)
T_LDMATRIX(t_ldmatrix_x4_trans, "x4.trans", 4, "{%0, %1, %2, %3}", R4)
// The address as an offset in shared memory, in a 32-bit register (what CUTLASS does), with a constant
// added in the operand.
KERNEL(t_ldmatrix_x4_shared32) {
    FILL_SHARED
    unsigned long long generic = (unsigned long long)((const char*)memory + in[GI]), offset;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(offset) : "l"(generic));
    unsigned start = (unsigned)offset - 16, r[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4+16];" : R4 : "r"(start));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = r[i];
}
KERNEL(t_ldmatrix_x2_trans_shared64) {
    FILL_SHARED
    unsigned long long generic = (unsigned long long)((const char*)memory + in[GI]), offset;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(offset) : "l"(generic));
    unsigned r[2];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 {%0, %1}, [%2];" : R2 : "l"(offset));
    out[GI * 2] = r[0], out[GI * 2 + 1] = r[1];
}

// ---- Turing: the sequences llama.cpp builds from these instructions -----------------------------------
// A 16x16 product with f32 accumulators in four instructions: operands used twice, accumulators chained.
// in: a0..a3 b0..b3 d0..d7, out: d0..d7.
KERNEL(t_chain_f32) {
    const unsigned* x = in + GI * 16;
    unsigned a[4] = {x[0], x[1], x[2], x[3]}, b[4] = {x[4], x[5], x[6], x[7]}, d[8];
    for (int i = 0; i < 8; i++) d[i] = x[8 + i];
#define MMA_F32(d0, a0, b0)                                                                                      \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};" \
                 : "+r"(d[d0]), "+r"(d[d0 + 1]), "+r"(d[d0 + 2]), "+r"(d[d0 + 3]) : "r"(a[a0]), "r"(a[a0 + 1]), "r"(b[b0]))
    MMA_F32(0, 0, 0);
    MMA_F32(0, 2, 2);
    MMA_F32(4, 0, 1);
    MMA_F32(4, 2, 3);
    for (int i = 0; i < 8; i++) out[GI * 8 + i] = d[i];
}
// The same with f16 accumulators. in: a0..a3 b0..b3 d0..d3, out: d0..d3.
KERNEL(t_chain_f16) {
    const unsigned* x = in + GI * 12;
    unsigned a[4] = {x[0], x[1], x[2], x[3]}, b[4] = {x[4], x[5], x[6], x[7]}, d[4] = {x[8], x[9], x[10], x[11]};
#define MMA_F16(d0, a0, b0)                                                                         \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};" \
                 : "+r"(d[d0]), "+r"(d[d0 + 1]) : "r"(a[a0]), "r"(a[a0 + 1]), "r"(b[b0]))
    MMA_F16(0, 0, 0);
    MMA_F16(0, 2, 2);
    MMA_F16(2, 0, 1);
    MMA_F16(2, 2, 3);
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// A 16x8x32 integer product in four instructions. in: a0..a3 b0 b1 d0..d3, out: d0..d3.
KERNEL(t_chain_s8) {
    const unsigned* x = in + GI * 10;
    unsigned a[4] = {x[0], x[1], x[2], x[3]}, b[2] = {x[4], x[5]}, d[4] = {x[6], x[7], x[8], x[9]};
#define MMA_S8(d0, a0, b0)                                                                         \
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" \
                 : "+r"(d[d0]), "+r"(d[d0 + 1]) : "r"(a[a0]), "r"(b[b0]))
    MMA_S8(0, 0, 0);
    MMA_S8(2, 1, 0);
    MMA_S8(0, 2, 1);
    MMA_S8(2, 3, 1);
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// Operands that change between instructions and across the iterations of a loop whose trip count is
// only known at run time (aux[0]); the transposed B register comes from movmatrix.
// in: a0..a3 b0..b3 f0..f3 h0 h1 n0 n1 (16 words), out: f0..f3 h0 h1 n0 n1 (8 words).
KERNEL(t_loop) {
    const unsigned* x = in + GI * 16;
    unsigned a[4] = {x[0], x[1], x[2], x[3]}, b[4] = {x[4], x[5], x[6], x[7]};
    unsigned f[4] = {x[8], x[9], x[10], x[11]}, h[2] = {x[12], x[13]}, n[2] = {x[14], x[15]};
    const int steps = aux[0];
    for (int step = 0; step < steps; step++) {
        unsigned a0 = a[2 * (step & 1)], a1 = a[2 * (step & 1) + 1], bb = b[step & 3], bt;
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                     : "+r"(f[0]), "+r"(f[1]), "+r"(f[2]), "+r"(f[3]) : "r"(a0), "r"(a1), "r"(bb));
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(bt) : "r"(bb));
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
                     : "+r"(h[0]), "+r"(h[1]) : "r"(a0), "r"(a1), "r"(bt));
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                     : "+r"(n[0]), "+r"(n[1]) : "r"(a0), "r"(bb));
        // The same register names get new values: nothing derived from the old ones may be reused.
        a[0] ^= a[2], a[2] ^= a[0], a[0] ^= a[2];
        b[step & 3] = bt;
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                     : "+r"(n[0]), "+r"(n[1]) : "r"(a[0]), "r"(b[step & 3]));
    }
    unsigned* y = out + GI * 8;
    y[0] = f[0], y[1] = f[1], y[2] = f[2], y[3] = f[3], y[4] = h[0], y[5] = h[1], y[6] = n[0], y[7] = n[1];
}
// Many instructions in one basic block: the rewriting ends the block now and then (see crates/vf-ptx/src/lower/).
// in: a0 a1 b f0..f3, out: f0..f3.
KERNEL(t_long) {
    const unsigned* x = in + GI * 7;
    unsigned a0 = x[0], a1 = x[1], b = x[2], f[4] = {x[3], x[4], x[5], x[6]};
#pragma unroll
    for (int step = 0; step < 12; step++) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                     : "+r"(f[0]), "+r"(f[1]), "+r"(f[2]), "+r"(f[3]) : "r"(a0), "r"(a1), "r"(b));
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(b) : "r"(b));
    }
    unsigned* y = out + GI * 4;
    y[0] = f[0], y[1] = f[1], y[2] = f[2], y[3] = f[3];
}

// ---- Turing: products whose operands come from shared memory ---------------------------------------
// What llama.cpp's kernels do: ldmatrix loads the operands of the products. The rewriting then has
// ldmatrix load them in the layout its Volta sequence takes, instead of shuffling them between threads
// (crates/vf-ptx/src/lower/). aux: 1024 words copied to shared memory as they are; in: byte offsets of matrix rows as
// for the ldmatrix kernels (the harness also gives offsets that are not multiples of 16).
#define FILL_SHARED_SAME                                                               \
    __shared__ unsigned memory[SHARED_WORDS];                                          \
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i];   \
    __syncthreads();
#define ROWS(word) ((const char*)memory + x[word])
// Both operands from plain ldmatrix, f32 accumulators: the K.Q product of flash attention.
// in: rows of A, rows of B, d0..d3; out: d0..d3.
KERNEL(t_ld_kq) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 6;
    unsigned a[4], b[2], d[4] = {x[2], x[3], x[4], x[5]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(ROWS(0)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(b[0]), "=r"(b[1]) : "l"(ROWS(1)));
    MMA_F32(0, 0, 0);
    MMA_F32(0, 2, 1);
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// B from ldmatrix.trans, A in registers, f16 accumulators: the V.KQ product of flash attention.
// in: a0..a3, rows of B, d0 d1; out: d0 d1.
KERNEL(t_ld_vkq) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 7;
    unsigned a[4] = {x[0], x[1], x[2], x[3]}, b[2], d[2] = {x[5], x[6]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];" : "=r"(b[0]), "=r"(b[1]) : "l"(ROWS(4)));
    MMA_F16(0, 0, 0);
    MMA_F16(0, 2, 1);
    out[GI * 2] = d[0], out[GI * 2 + 1] = d[1];
}
// A of an integer product from ldmatrix, B in registers: the quantized matrix product.
// in: rows of A, b0 b1, d0..d3; out: d0..d3.
KERNEL(t_ld_s8) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 7;
    unsigned a[2], b[2] = {x[1], x[2]}, d[4] = {x[3], x[4], x[5], x[6]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(a[0]), "=r"(a[1]) : "l"(ROWS(0)));
    MMA_S8(0, 0, 0);
    MMA_S8(2, 1, 1);
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
// The registers of one ldmatrix read three ways: by a product that has its operands loaded, by one
// that takes them as they are (its B register is not loaded) and by a store.
// in: rows of A, rows of B, another B register, d0..d3; out: d0..d3, a0, b0.
KERNEL(t_ld_mixed) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 7;
    unsigned a[2], b[2], d[4] = {x[3], x[4], x[5], x[6]};
    b[1] = x[2];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(a[0]), "=r"(a[1]) : "l"(ROWS(0)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];" : "=r"(b[0]) : "l"(ROWS(1)));
    MMA_F32(0, 0, 0);
    MMA_F32(0, 0, 1);
    unsigned* y = out + GI * 6;
    y[0] = d[0], y[1] = d[1], y[2] = d[2], y[3] = d[3], y[4] = a[0], y[5] = b[0];
}
// B loaded once before a loop whose trip count is only known at run time, A loaded in the loop: the
// registers that hold the operands in Volta layout live across the iterations.
// in: rows of A, rows of B, trip count, d0..d3; out: d0..d3.
KERNEL(t_ld_loop) {
    FILL_SHARED_SAME
    const unsigned* x = in + GI * 7;
    const int steps = x[2];
    unsigned a[2], b[2], d[4] = {x[3], x[4], x[5], x[6]};
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(b[0]), "=r"(b[1]) : "l"(ROWS(1)));
    for (int step = 0; step < steps; step++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(a[0]), "=r"(a[1]) : "l"(ROWS(0)));
        MMA_F32(0, 0, 0);
        MMA_F32(0, 0, 1);
    }
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
