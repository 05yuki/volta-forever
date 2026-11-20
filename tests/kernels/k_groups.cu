// Kernels for testing GROUP patterns in the translator: one ldmatrix.x4 for A
// (4 registers that are two A operands, registers 0-1 and 2-3), one ldmatrix.x4
// for B (4 registers that are four B operands), and the products of each A by each
// B into separate f32 accumulators.  The translator will treat a GROUP specially,
// keeping the accumulators in another arrangement between products and converting
// them before anything else reads them.  These tests exercise every way of entering,
// leaving and interrupting such a chain.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o tests/kernels/k_groups.135.sm75.cubin tests/kernels/k_groups.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -ptx   -o tests/kernels/k_groups.135.sm75.ptx   tests/kernels/k_groups.cu

#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)
#define MMA_F32 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
#define MMA_F16 "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
#define SHARED_WORDS 1024

#define FILL_SHARED                                                                \
    __shared__ unsigned memory[SHARED_WORDS];                                      \
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i]; \
    __syncthreads();

// Macros for ldmatrix loads from shared memory.
#define LOAD_A4(a, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"         \
        : "=r"((a)[0]),"=r"((a)[1]),"=r"((a)[2]),"=r"((a)[3]) : "l"(addr))
#define LOAD_B4(b, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"         \
        : "=r"((b)[0]),"=r"((b)[1]),"=r"((b)[2]),"=r"((b)[3]) : "l"(addr))
#define LOAD_B4T(b, addr)                                                          \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3}, [%4];"   \
        : "=r"((b)[0]),"=r"((b)[1]),"=r"((b)[2]),"=r"((b)[3]) : "l"(addr))
#define LOAD_B1(b, addr)                                                           \
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];" : "=r"(b) : "l"(addr))

// All 8 products of a GROUP in one asm block so the compiler keeps them in order.
// d[0..31]: accumulators (8 sets of 4), a[0..3]: A registers, b[0..3]: B registers.
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

// First 4 products (A0 x B0..B3) and last 4 (A1 x B0..B3) as separate blocks,
// so loads for the next iteration can be placed between them.
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

// All 8 products with f16 accumulators: 16 output registers.
#define GROUP_ALL_F16(d, a, b) asm volatile(                                       \
    MMA_F16 "{%0,%1}, {%16,%17}, {%20}, {%0,%1};\n\t"                             \
    MMA_F16 "{%2,%3}, {%16,%17}, {%21}, {%2,%3};\n\t"                             \
    MMA_F16 "{%4,%5}, {%16,%17}, {%22}, {%4,%5};\n\t"                             \
    MMA_F16 "{%6,%7}, {%16,%17}, {%23}, {%6,%7};\n\t"                             \
    MMA_F16 "{%8,%9}, {%18,%19}, {%20}, {%8,%9};\n\t"                             \
    MMA_F16 "{%10,%11}, {%18,%19}, {%21}, {%10,%11};\n\t"                         \
    MMA_F16 "{%12,%13}, {%18,%19}, {%22}, {%12,%13};\n\t"                         \
    MMA_F16 "{%14,%15}, {%18,%19}, {%23}, {%14,%15};"                             \
    : "+r"((d)[0]),  "+r"((d)[1]),  "+r"((d)[2]),  "+r"((d)[3]),                  \
      "+r"((d)[4]),  "+r"((d)[5]),  "+r"((d)[6]),  "+r"((d)[7]),                  \
      "+r"((d)[8]),  "+r"((d)[9]),  "+r"((d)[10]), "+r"((d)[11]),                 \
      "+r"((d)[12]), "+r"((d)[13]), "+r"((d)[14]), "+r"((d)[15])                  \
    : "r"((a)[0]), "r"((a)[1]), "r"((a)[2]), "r"((a)[3]),                         \
      "r"((b)[0]), "r"((b)[1]), "r"((b)[2]), "r"((b)[3]))

// Helpers
#define ADDR(x, i) ((const char*)memory + (x)[(i)])

static __device__ void scale32(unsigned* d, float s) {
    for (int i = 0; i < 32; i++) {
        float f;
        memcpy(&f, &d[i], 4);
        f *= s;
        memcpy(&d[i], &f, 4);
    }
}

// ---------------------------------------------------------------------------
// 1. g_once: one group, accumulators from zero, all written out.
//    in: o_a, o_b (2 words); out: 32 f32 accumulators.
KERNEL(g_once) {
    FILL_SHARED
    const unsigned* x = in + GI * 2;
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    LOAD_A4(a, ADDR(x, 0));
    LOAD_B4(b, ADDR(x, 1));
    GROUP_ALL(d, a, b);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 2. g_loop: group in a loop, software-pipelined (loads of next slice before last
//    products of current), accumulators zeroed, skip when count is 0, multiply by
//    0.5 after.  in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(g_loop) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], an[4], bn[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            if (step + 1 < steps) {
                LOAD_A4(an, ADDR(x, 1));
                LOAD_B4(bn, ADDR(x, 2));
            }
            GROUP_A1(d, a, b);
            if (step + 1 < steps) {
                for (int i = 0; i < 4; i++) a[i] = an[i], b[i] = bn[i];
            }
        }
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 3. g_trans: as g_loop with B loaded by ldmatrix.trans.
//    in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(g_trans) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], an[4], bn[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4T(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            if (step + 1 < steps) {
                LOAD_A4(an, ADDR(x, 1));
                LOAD_B4T(bn, ADDR(x, 2));
            }
            GROUP_A1(d, a, b);
            if (step + 1 < steps) {
                for (int i = 0; i < 4; i++) a[i] = an[i], b[i] = bn[i];
            }
        }
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 4. g_init: as g_loop with accumulators loaded from input (not zero).
//    in: count, o_a, o_b, d0..d31 (35 words); out: 32 f32.
KERNEL(g_init) {
    FILL_SHARED
    const unsigned* x = in + GI * 35;
    const int steps = x[0];
    unsigned a[4], b[4], an[4], bn[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = x[3 + i];
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            if (step + 1 < steps) {
                LOAD_A4(an, ADDR(x, 1));
                LOAD_B4(bn, ADDR(x, 2));
            }
            GROUP_A1(d, a, b);
            if (step + 1 < steps) {
                for (int i = 0; i < 4; i++) a[i] = an[i], b[i] = bn[i];
            }
        }
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 5. g_peek: as g_loop, with accumulators read and changed between products on
//    even turns (a uniform condition).  d[0] and d[16] are added to peek outputs;
//    d[0] is sign-flipped.  in: count, o_a, o_b (3 words); out: 34 f32.
KERNEL(g_peek) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    unsigned pk0 = 0, pk1 = 0;  // peek accumulators (f32 as bits)
    for (int i = 0; i < 32; i++) d[i] = 0;
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            if (step % 2 == 0) {
                // Peek: read d[0] and d[16], add to outputs, overwrite d[0]
                float f0, f16, s0, s1;
                memcpy(&f0, &d[0], 4);
                memcpy(&f16, &d[16], 4);
                memcpy(&s0, &pk0, 4);
                memcpy(&s1, &pk1, 4);
                s0 += f0;
                s1 += f16;
                memcpy(&pk0, &s0, 4);
                memcpy(&pk1, &s1, 4);
                d[0] ^= 0x80000000u;  // flip sign
            }
            if (step + 1 < steps) {
                LOAD_A4(a, ADDR(x, 1));
                LOAD_B4(b, ADDR(x, 2));
            }
            GROUP_A1(d, a, b);
        }
    }
    scale32(d, 0.5f);
    unsigned* y = out + GI * 34;
    for (int i = 0; i < 32; i++) y[i] = d[i];
    y[32] = pk0;
    y[33] = pk1;
}

// ---------------------------------------------------------------------------
// 6. g_partial: six of the eight products only, and one product alone on a B that
//    nothing else uses.  in: o_a, o_b, o_bx (3 words); out: 28 f32.
KERNEL(g_partial) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    unsigned a[4], b[4], bx, d[28];
    for (int i = 0; i < 28; i++) d[i] = 0;
    LOAD_A4(a, ADDR(x, 0));
    LOAD_B4(b, ADDR(x, 1));
    LOAD_B1(bx, ADDR(x, 2));
    // 6 of the 8 group products (A0 x B0..B3, A1 x B0, A1 x B1)
    asm volatile(
        MMA_F32 "{%0,%1,%2,%3}, {%24,%25}, {%28}, {%0,%1,%2,%3};\n\t"
        MMA_F32 "{%4,%5,%6,%7}, {%24,%25}, {%29}, {%4,%5,%6,%7};\n\t"
        MMA_F32 "{%8,%9,%10,%11}, {%24,%25}, {%30}, {%8,%9,%10,%11};\n\t"
        MMA_F32 "{%12,%13,%14,%15}, {%24,%25}, {%31}, {%12,%13,%14,%15};\n\t"
        MMA_F32 "{%16,%17,%18,%19}, {%26,%27}, {%28}, {%16,%17,%18,%19};\n\t"
        MMA_F32 "{%20,%21,%22,%23}, {%26,%27}, {%29}, {%20,%21,%22,%23};"
        : "+r"(d[0]),  "+r"(d[1]),  "+r"(d[2]),  "+r"(d[3]),
          "+r"(d[4]),  "+r"(d[5]),  "+r"(d[6]),  "+r"(d[7]),
          "+r"(d[8]),  "+r"(d[9]),  "+r"(d[10]), "+r"(d[11]),
          "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),
          "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
    // 1 isolated product: A0 x Bx (a B that nothing else uses)
    asm volatile(
        MMA_F32 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27])
        : "r"(a[0]), "r"(a[1]), "r"(bx));
    unsigned* y = out + GI * 28;
    for (int i = 0; i < 28; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 7. g_shared: a load whose result is ALSO read by an ordinary instruction
//    (written out), beside the products.  in: o_a, o_b (2 words); out: 36.
KERNEL(g_shared) {
    FILL_SHARED
    const unsigned* x = in + GI * 2;
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    LOAD_A4(a, ADDR(x, 0));
    LOAD_B4(b, ADDR(x, 1));
    GROUP_ALL(d, a, b);
    unsigned* y = out + GI * 36;
    for (int i = 0; i < 32; i++) y[i] = d[i];
    // Also write the raw A registers (the load read by an ordinary instruction)
    y[32] = a[0]; y[33] = a[1]; y[34] = a[2]; y[35] = a[3];
}

// ---------------------------------------------------------------------------
// 8. g_two: two loops one after the other on the same accumulator registers,
//    with a divergent branch between them (lane >= 16 multiplies by 2.0, the
//    others by 0.5).  in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(g_two) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    // First loop
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            GROUP_A1(d, a, b);
        }
    }
    // Divergent branch between loops
    unsigned lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    if (lane_id >= 16) {
        scale32(d, 2.0f);
    } else {
        scale32(d, 0.5f);
    }
    // Second loop
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            GROUP_A1(d, a, b);
        }
    }
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 9. g_blocks: a loop whose body has two blocks of products: on even turns
//    (a uniform condition) an extra pair of loads feeds another group of 8
//    products into the same accumulators.
//    in: count, o_a, o_b, o_a2, o_b2 (5 words); out: 32 f32.
KERNEL(g_blocks) {
    FILL_SHARED
    const unsigned* x = in + GI * 5;
    const int steps = x[0];
    unsigned a[4], b[4], a2[4], b2[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
#pragma unroll 1
    for (int step = 0; step < steps; step++) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
        GROUP_A0(d, a, b);
        GROUP_A1(d, a, b);
        if (step % 2 == 0) {
            LOAD_A4(a2, ADDR(x, 3));
            LOAD_B4(b2, ADDR(x, 4));
            GROUP_A0(d, a2, b2);
            GROUP_A1(d, a2, b2);
        }
    }
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 10. g_after: after the loop, the first read of the accumulators is inside a
//     divergent branch (lane < 16 multiplies by 2.0 first), the others read
//     them later unchanged.
//     in: count, o_a, o_b (3 words); out: 32 f32.
KERNEL(g_after) {
    FILL_SHARED
    const unsigned* x = in + GI * 3;
    const int steps = x[0];
    unsigned a[4], b[4], d[32];
    for (int i = 0; i < 32; i++) d[i] = 0;
    if (steps > 0) {
        LOAD_A4(a, ADDR(x, 1));
        LOAD_B4(b, ADDR(x, 2));
#pragma unroll 1
        for (int step = 0; step < steps; step++) {
            GROUP_A0(d, a, b);
            GROUP_A1(d, a, b);
        }
    }
    unsigned lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    if (lane_id < 16) {
        scale32(d, 2.0f);
    }
    unsigned* y = out + GI * 32;
    for (int i = 0; i < 32; i++) y[i] = d[i];
}

// ---------------------------------------------------------------------------
// 11. g_sixteen: products with f16 accumulators, one group.
//     in: o_a, o_b (2 words); out: 16 f16 pairs.
KERNEL(g_sixteen) {
    FILL_SHARED
    const unsigned* x = in + GI * 2;
    unsigned a[4], b[4], d[16];
    for (int i = 0; i < 16; i++) d[i] = 0;
    LOAD_A4(a, ADDR(x, 0));
    LOAD_B4(b, ADDR(x, 1));
    GROUP_ALL_F16(d, a, b);
    unsigned* y = out + GI * 16;
    for (int i = 0; i < 16; i++) y[i] = d[i];
}
