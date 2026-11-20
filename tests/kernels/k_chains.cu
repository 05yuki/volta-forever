// k_chains.cu: test kernels for the chain form of the f32 products (crates/vf-ptx/src/lower/,
// `mma_f32_chain`): one Volta product for a Turing one while the accumulator stays split in two
// partial sums, given back to the registers of the program when the chain ends.
//
// Build: toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -ptx -o k_chains.135.sm75.ptx k_chains.cu

#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// MMA helper: one m16n8k8 f32 product accumulating in place.
#define MMA(d, a0, a1, b) \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 " \
        "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" \
        :"+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]) :"r"(a0),"r"(a1),"r"(b))

// t_chain_end_read: chain of 4, then add.f32 reading the accumulator, then chain of 4.
// Tests that chain_end is called before the add.
// in: a0 a1 b0 b1 b2 b3 c0..c3 scalar (11 words). out: d0..d3 (4 words).
KERNEL(t_chain_end_read) {
    const unsigned* x = in + GI * 11;
    unsigned a0 = x[0], a1 = x[1];
    unsigned b[4] = {x[2], x[3], x[4], x[5]};
    unsigned d[4] = {x[6], x[7], x[8], x[9]};
    unsigned scalar = x[10];

    MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
    MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);

    // Non-product read of the accumulator
    asm volatile("add.f32 %0, %0, %1;" :"+r"(d[0]) :"r"(scalar));
    asm volatile("add.f32 %0, %0, %1;" :"+r"(d[1]) :"r"(scalar));
    asm volatile("add.f32 %0, %0, %1;" :"+r"(d[2]) :"r"(scalar));
    asm volatile("add.f32 %0, %0, %1;" :"+r"(d[3]) :"r"(scalar));

    MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
    MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);

    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// t_chain_carry: two chains of 4 on the same accumulator with a non-touching statement
// (a mul of a separate register) between them. The split form carries across.
// in: a0 a1 b0 b1 b2 b3 c0..c3 extra (11 words). out: d0..d3 (4 words).
KERNEL(t_chain_carry) {
    const unsigned* x = in + GI * 11;
    unsigned a0 = x[0], a1 = x[1];
    unsigned b[4] = {x[2], x[3], x[4], x[5]};
    unsigned d[4] = {x[6], x[7], x[8], x[9]};
    unsigned extra = x[10];

    MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
    MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);

    // Statement that does NOT touch the accumulator d[0..3]
    asm volatile("mul.f32 %0, %0, %0;" :"+r"(extra));

    MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
    MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);

    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// t_chain_loop: a loop of 4 turns, each turn 4 products on the same accumulator. The compiler
// knows the count and writes the 16 products in a line, without a branch: a chain of 16.
// (t_chain_back is the loop with a back edge.)
// in: a0 a1 b0 b1 b2 b3 c0..c3 (10 words). out: d0..d3 (4 words).
KERNEL(t_chain_loop) {
    const unsigned* x = in + GI * 10;
    unsigned a0 = x[0], a1 = x[1];
    unsigned b[4] = {x[2], x[3], x[4], x[5]};
    unsigned d[4] = {x[6], x[7], x[8], x[9]};

    for (int turn = 0; turn < 4; turn++) {
        MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
        MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);
    }

    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// t_chain_back: a loop whose number of turns is read from memory, so that the compiler keeps
// its back edge, each turn a chain of 4 products on one accumulator; and after the loop a
// product on another accumulator before the first one is read. The lanes that take the back
// edge must find the accumulator in the program's layout at the top of the loop: the chain is
// ended before the branch, not at the next statement that reads it (which is after the loop,
// behind a statement the lowering rewrites).
// in: a0 a1 b0 b1 b2 b3 c0..c3 e0..e3 (14 words). aux[0]: the number of turns. out: d0..d3 e0..e3.
KERNEL(t_chain_back) {
    const unsigned* x = in + GI * 14;
    unsigned a0 = x[0], a1 = x[1];
    unsigned b[4] = {x[2], x[3], x[4], x[5]};
    unsigned d[4] = {x[6], x[7], x[8], x[9]};
    unsigned e[4] = {x[10], x[11], x[12], x[13]};
    int turns = aux[0];

    // (A loop entered without a test: the code after it is then reached from its last branch
    // only and carries no label, so that nothing but the branch itself tells the lowering that
    // lanes leave the straight line there.)
    int turn = 0;
    do {
        MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
        MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);
    } while (++turn < turns);
    MMA(e, a0, a1, b[0]);

    for (int i = 0; i < 4; i++) out[GI * 8 + i] = d[i];
    for (int i = 0; i < 4; i++) out[GI * 8 + 4 + i] = e[i];
}

// t_chain_exit: four products on one accumulator, a way out that the whole warp takes or not
// (the same word of memory decides it in every thread), four more products on it. The code
// between the branch and its target is entered from the branch's line only: it has no label, and
// its first statement is a product of the same chain. The lanes that leave must find the
// accumulator in the program's layout where they go.
// in: a0 a1 b0 b1 b2 b3 c0..c3 (10 words). aux[1]: not zero to leave after four. out: d0..d3.
KERNEL(t_chain_exit) {
    const unsigned* x = in + GI * 10;
    unsigned a0 = x[0], a1 = x[1];
    unsigned b[4] = {x[2], x[3], x[4], x[5]};
    unsigned d[4] = {x[6], x[7], x[8], x[9]};

    MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
    MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);
    if (aux[1] == 0) {
        MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
        MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);
    }

    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// t_chain_zero: an accumulator that starts from zero (set through another register, as the
// compiler writes llama.cpp's), six products. Nothing is fetched from the other lanes to split it.
// in: a0 a1 b0 b1 b2 b3 (6 words). out: d0..d3.
KERNEL(t_chain_zero) {
    const unsigned* x = in + GI * 6;
    unsigned a0 = x[0], a1 = x[1];
    unsigned b[4] = {x[2], x[3], x[4], x[5]};
    unsigned d[4] = {0, 0, 0, 0};

    MMA(d, a0, a1, b[0]); MMA(d, a0, a1, b[1]);
    MMA(d, a0, a1, b[2]); MMA(d, a0, a1, b[3]);
    MMA(d, a0, a1, b[1]); MMA(d, a0, a1, b[2]);

    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

// t_chain_ld: what llama.cpp's K.Q product is made of. The operands come from shared memory by
// ldmatrix (two pairs of A in one, four registers of B in another), two accumulators start from
// zero and take four products each in turn: two chains open at once, whose operands are loaded
// by the ldmatrix in the layout of the chain's single instruction and nowhere else.
// aux: 1024 words copied to shared memory. in: rows of A, rows of B (byte offsets, 2 words).
// out: d0..d3 e0..e3.
#define SHARED_WORDS 1024
KERNEL(t_chain_ld) {
    __shared__ unsigned memory[SHARED_WORDS];
    for (int i = threadIdx.x; i < SHARED_WORDS; i += blockDim.x) memory[i] = aux[i];
    __syncthreads();
    const unsigned* x = in + GI * 2;
    unsigned a[4], b[4];
    unsigned d[4] = {0, 0, 0, 0}, e[4] = {0, 0, 0, 0};
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"((const char*)memory + x[0]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "l"((const char*)memory + x[1]));

    MMA(d, a[0], a[1], b[0]); MMA(d, a[2], a[3], b[2]);
    MMA(e, a[0], a[1], b[1]); MMA(e, a[2], a[3], b[3]);
    MMA(d, a[0], a[1], b[2]); MMA(d, a[2], a[3], b[0]);
    MMA(e, a[0], a[1], b[3]); MMA(e, a[2], a[3], b[1]);

    for (int i = 0; i < 4; i++) out[GI * 8 + i] = d[i];
    for (int i = 0; i < 4; i++) out[GI * 8 + 4 + i] = e[i];
}
