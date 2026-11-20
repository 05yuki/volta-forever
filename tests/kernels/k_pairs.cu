// Kernels for testing PAIRS in the translator: each kernel exercises a feature
// of the "pairs of products on whole rows" arrangement (docs/pairs.md) and is written
// so that the compiler produces unconditional LDSM loads that reach the products
// directly, without predicated loads or SEL instructions.
//
// Build:
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o tests/kernels/k_pairs.135.sm75.cubin tests/kernels/k_pairs.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -ptx   -o tests/kernels/k_pairs.135.sm75.ptx   tests/kernels/k_pairs.cu

#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)
#define MMA_F32 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
#define SHARED_WORDS 1024

#define FILL_SHARED                                                                \
    __shared__ unsigned memory[SHARED_WORDS];                                      \
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i]; \
    __syncthreads();

#define LOAD_A4(a, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"         \
        : "=r"((a)[0]),"=r"((a)[1]),"=r"((a)[2]),"=r"((a)[3]) : "l"(addr))
#define LOAD_B4(b, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"         \
        : "=r"((b)[0]),"=r"((b)[1]),"=r"((b)[2]),"=r"((b)[3]) : "l"(addr))
#define LOAD_B4T(b, addr)                                                          \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"   \
        : "=r"((b)[0]),"=r"((b)[1]),"=r"((b)[2]),"=r"((b)[3]) : "l"(addr))

#define ADDR(x, i) ((const char*)memory + (x)[(i)])

// 4 products with A0: A0 x B0..B3, each its own accumulator.
#define GROUP_A0(d, a, b) asm volatile(                                            \
    MMA_F32 "{%0,%1,%2,%3}, {%16,%17}, {%18}, {%0,%1,%2,%3};\n\t"                 \
    MMA_F32 "{%4,%5,%6,%7}, {%16,%17}, {%19}, {%4,%5,%6,%7};\n\t"                 \
    MMA_F32 "{%8,%9,%10,%11}, {%16,%17}, {%20}, {%8,%9,%10,%11};\n\t"             \
    MMA_F32 "{%12,%13,%14,%15}, {%16,%17}, {%21}, {%12,%13,%14,%15};"             \
    : "+r"((d)[0]),  "+r"((d)[1]),  "+r"((d)[2]),  "+r"((d)[3]),                  \
      "+r"((d)[4]),  "+r"((d)[5]),  "+r"((d)[6]),  "+r"((d)[7]),                  \
      "+r"((d)[8]),  "+r"((d)[9]),  "+r"((d)[10]), "+r"((d)[11]),                 \
      "+r"((d)[12]), "+r"((d)[13]), "+r"((d)[14]), "+r"((d)[15])                  \
    : "r"((a)[0]), "r"((a)[1]), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))

#define GROUP_A1(d, a, b) asm volatile(                                            \
    MMA_F32 "{%0,%1,%2,%3}, {%16,%17}, {%18}, {%0,%1,%2,%3};\n\t"                 \
    MMA_F32 "{%4,%5,%6,%7}, {%16,%17}, {%19}, {%4,%5,%6,%7};\n\t"                 \
    MMA_F32 "{%8,%9,%10,%11}, {%16,%17}, {%20}, {%8,%9,%10,%11};\n\t"             \
    MMA_F32 "{%12,%13,%14,%15}, {%16,%17}, {%21}, {%12,%13,%14,%15};"             \
    : "+r"((d)[16]), "+r"((d)[17]), "+r"((d)[18]), "+r"((d)[19]),                 \
      "+r"((d)[20]), "+r"((d)[21]), "+r"((d)[22]), "+r"((d)[23]),                 \
      "+r"((d)[24]), "+r"((d)[25]), "+r"((d)[26]), "+r"((d)[27]),                 \
      "+r"((d)[28]), "+r"((d)[29]), "+r"((d)[30]), "+r"((d)[31])                  \
    : "r"((a)[2]), "r"((a)[3]), "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))

// All 8 products in one block.
#define GROUP_ALL(d, a, b) asm volatile(                                           \
    MMA_F32 "{%0,%1,%2,%3}, {%32,%33}, {%36}, {%0,%1,%2,%3};\n\t"                 \
    MMA_F32 "{%4,%5,%6,%7}, {%32,%33}, {%37}, {%4,%5,%6,%7};\n\t"                 \
    MMA_F32 "{%8,%9,%10,%11}, {%32,%33}, {%38}, {%8,%9,%10,%11};\n\t"             \
    MMA_F32 "{%12,%13,%14,%15}, {%32,%33}, {%39}, {%12,%13,%14,%15};\n\t"         \
    MMA_F32 "{%16,%17,%18,%19}, {%34,%35}, {%36}, {%16,%17,%18,%19};\n\t"         \
    MMA_F32 "{%20,%21,%22,%23}, {%34,%35}, {%37}, {%20,%21,%22,%23};\n\t"         \
    MMA_F32 "{%24,%25,%26,%27}, {%34,%35}, {%38}, {%24,%25,%26,%27};\n\t"         \
    MMA_F32 "{%28,%29,%30,%31}, {%34,%35}, {%39}, {%28,%29,%30,%31};"             \
    : "+r"((d)[0]),  "+r"((d)[1]),  "+r"((d)[2]),  "+r"((d)[3]),                  \
      "+r"((d)[4]),  "+r"((d)[5]),  "+r"((d)[6]),  "+r"((d)[7]),                  \
      "+r"((d)[8]),  "+r"((d)[9]),  "+r"((d)[10]), "+r"((d)[11]),                 \
      "+r"((d)[12]), "+r"((d)[13]), "+r"((d)[14]), "+r"((d)[15]),                 \
      "+r"((d)[16]), "+r"((d)[17]), "+r"((d)[18]), "+r"((d)[19]),                 \
      "+r"((d)[20]), "+r"((d)[21]), "+r"((d)[22]), "+r"((d)[23]),                 \
      "+r"((d)[24]), "+r"((d)[25]), "+r"((d)[26]), "+r"((d)[27]),                 \
      "+r"((d)[28]), "+r"((d)[29]), "+r"((d)[30]), "+r"((d)[31])                  \
    : "r"((a)[0]), "r"((a)[1]), "r"((a)[2]), "r"((a)[3]),                         \
      "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))

static __device__ void scale32(unsigned* d, float s) {
    for (int i = 0; i < 32; i++) {
        float f;
        memcpy(&f, &d[i], 4);
        f *= s;
        memcpy(&d[i], &f, 4);
    }
}

// ---------------------------------------------------------------------------
// 1. p_basic: a loop with unconditional loads of A and B at each turn, then
//    the eight products of two A by four B.  The loads are always in the loop
//    body, no conditional.  in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(p_basic) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 2. p_trans: the same with B loaded transposed (ldmatrix ... .trans).
//    in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(p_trans) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4T(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 3. p_pipe: loads of the next turn placed between the products of this turn,
//    with no condition.  Two register sets (a0/b0 and a1/b1) are double-
//    buffered: even turns use a0/b0 and load a1/b1 between products; odd
//    turns use a1/b1 and load a0/b0.  The loop is manually unrolled by 2 so
//    that the compiler does not need MOV copies to swap registers.
//    in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(p_pipe) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a0[4], b0[4], a1[4], b1[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    if (steps > 0) {
        LOAD_A4(a0, ADDR(x, 1));
        LOAD_B4(b0, ADDR(x, 2));
        int step = 0;
#pragma unroll 1
        for (; step + 1 < steps; step += 2) {
            // Even turn: products with a0/b0, load a1/b1 between
            GROUP_A0(d, a0, b0);
            LOAD_A4(a1, ADDR(x, 1));
            LOAD_B4(b1, ADDR(x, 2));
            GROUP_A1(d, a0, b0);
            // Odd turn: products with a1/b1, load a0/b0 between
            GROUP_A0(d, a1, b1);
            LOAD_A4(a0, ADDR(x, 1));
            LOAD_B4(b0, ADDR(x, 2));
            GROUP_A1(d, a1, b1);
        }
        // Odd remainder: one more turn with a0/b0 (no intermediate loads)
        if (step < steps) {
            GROUP_A0(d, a0, b0);
            GROUP_A1(d, a0, b0);
        }
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 4. p_init: accumulators that come from the input and not from zero.
//    in: count, o_a, o_b, d0..d31 (35 words); out: 32 f32.
KERNEL(p_init) {
    FILL_SHARED
    const unsigned* x = in + GI * 35;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = x[3 + i];
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 5. p_peek: accumulators read and changed between turns under a condition that
//    is the same for all threads (every other turn), by ordinary instructions.
//    d[0] and d[16] are read and added to peek outputs; d[0] is sign-flipped.
//    The peek is between the two halves of products (GROUP_A0 and GROUP_A1),
//    so the accumulators must convert T<->Q.
//    in: count, o_a, o_b (3 words); out: 34 f32.
KERNEL(p_peek) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    unsigned pk0 = 0, pk1 = 0;
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        if (step % 2 == 0) {
            float f0, f16, s0, s1;
            memcpy(&f0, &d[0], 4);
            memcpy(&f16, &d[16], 4);
            memcpy(&s0, &pk0, 4);
            memcpy(&s1, &pk1, 4);
            s0 += f0;
            s1 += f16;
            memcpy(&pk0, &s0, 4);
            memcpy(&pk1, &s1, 4);
            d[0] ^= 0x80000000u;
        }
        GROUP_A1(d, a, b);
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 34;
    for (int i = 0; i < 32; i++) y[i] = d[i];
    y[32] = pk0;
    y[33] = pk1;
}

// ---------------------------------------------------------------------------
// 6. p_after: after the loop, a read of the accumulators by some lanes only
//    (lane < 16 multiplies by 2.0, the others leave them as is); and two loops
//    one after the other with a divergent branch between them.
//    in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(p_after) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    // First loop
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    // Divergent branch between loops: lane < 16 doubles, others halve
    unsigned lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    if (lane_id < 16) {
        scale32(d, 2.0f);
    } else {
        scale32(d, 0.5f);
    }
    // Second loop
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 7. p_shared: inside a loop, a load whose registers are also read by an
//    ordinary instruction (stored to output).  The A load's registers are
//    written out at each turn, so the load is "also".
//    in: count, o_a, o_b (3 words); out: 36 f32.
KERNEL(p_shared) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    unsigned asum[4] = {0, 0, 0, 0};
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        // Read the A registers with ordinary instructions: XOR into asum.
        // This makes the A load "also" (read by non-product instructions).
        asum[0] ^= a[0];
        asum[1] ^= a[1];
        asum[2] ^= a[2];
        asum[3] ^= a[3];
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 36;
    for (int i = 0; i < 32; i++) y[i] = d[i];
    y[32] = asum[0]; y[33] = asum[1]; y[34] = asum[2]; y[35] = asum[3];
}

// ---------------------------------------------------------------------------
// 8. p_multi: two loops with different A and B registers (four loads total),
//    enough that the planner makes destinations share registers.
//    Loop 1: loads A1, B1, products into d[0..31].
//    Loop 2: loads A2, B2, products into the same d[0..31].
//    in: count, o_a1, o_b1, o_a2, o_b2 (5 words); out: 32 f32.
KERNEL(p_multi) {
    FILL_SHARED
    const unsigned* x = in + GI * 5;
    const int steps = x[0];
    unsigned a1[4], b1[4], a2[4], b2[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    // First loop
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a1, ADDR(x, 1));
        LOAD_B4(b1, ADDR(x, 2));
        GROUP_A0(d, a1, b1);
        GROUP_A1(d, a1, b1);
    }
    // Second loop with different addresses (and thus potentially different loads)
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a2, ADDR(x, 3));
        LOAD_B4(b2, ADDR(x, 4));
        GROUP_A0(d, a2, b2);
        GROUP_A1(d, a2, b2);
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 9a. p_call: a __noinline__ device function called between two loops of
//    products.  The call requires all registers to be in state T.
//    in: count, o_a, o_b (3 words); out: 32 f32.

__device__ __noinline__ void p_call_scale(unsigned* d, float s) {
    for (int i = 0; i < 32; i++) {
        float f;
        memcpy(&f, &d[i], 4);
        f *= s;
        memcpy(&d[i], &f, 4);
    }
}

KERNEL(p_call) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    // First loop
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    // Non-inline call between the loops
    p_call_scale(d, 0.5f);
    // Second loop
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
    }
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 9b. p_callee: a __noinline__ device function that has the products inside it.
//    in: o_a, o_b (2 words); out: 32 f32.

__device__ __noinline__ void p_callee_work(const unsigned* memory, const unsigned* x, unsigned* d) {
    unsigned a[4], b[4];
    LOAD_A4(a, ADDR(x, 0));
    LOAD_B4(b, ADDR(x, 1));
    GROUP_A0(d, a, b);
    GROUP_A1(d, a, b);
}

KERNEL(p_callee) {
    FILL_SHARED
    const unsigned* x = in + GI * 2;
    unsigned d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    p_callee_work(memory, x, d);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 10a. p_uniform: a uniform conditional branch that the design says is whole
//     (section 7, "uniform"): the loop contains a branch taken by all threads
//     or by none, with loads and products on both ways.  On even steps, an
//     extra product is computed; on odd steps, the accumulators are halved.
//     The branch condition (step % 2) is uniform.
//     in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(p_uniform) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
        if (step % 2 == 0) {
            // Extra set of products on even steps
            LOAD_A4(a, ADDR(x, 1));
            LOAD_B4(b, ADDR(x, 2));
            GROUP_A0(d, a, b);
            GROUP_A1(d, a, b);
        }
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 10b. p_lead: a conversion placed at a "lead" position (section 7): the
//     products are after a syncthreads, so the conversion is before them and
//     code that is straight-line to the aligned instruction.
//     in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(p_lead) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
        // A syncthreads in the loop: re-synchronize, forcing conversions
        __syncthreads();
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}
