// Kernels with several products on one register of B, for tcrun.c: the translation of machine
// code gathers the elements of B for a product and keeps them for the next products on the same
// register, as long as nothing writes it. Each kernel has its products in one block
// of inline PTX, so that the compiler leaves them in this order, on these registers.
//
// t_kept_f32, t_kept_f16: two products with the same B and two A, accumulators in floats or in
//     halves: the second takes what the first gathered.
// t_kept_again: the same two, then the register of B is written (its signs turned by those of a
//     result of the first product: values stay ordinary numbers), then two products more: the
//     third must gather again, the fourth takes what the third gathered.
// t_kept_chain: two products with the same B that add into the same accumulators: the second
//     takes what the first gathered, and its first step reads what the first product's last
//     instructions have just written.
// t_kept_zero: two products with the same B and no accumulators to add to: the instruction has
//     the zero register in their place.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_kept.135.sm75.cubin k_kept.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -ptx -o k_kept.135.sm75.ptx k_kept.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)
#define MMA_F32 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
#define MMA_F16 "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "

// in: a0 a1 (A of the first product), a2 a3 (A of the second), b, d0..d3, e0..e3 (13 words);
// out: d0..d3 (the first product), e0..e3 (the second).
KERNEL(t_kept_f32) {
    const unsigned* x = in + GI * 13;
    unsigned a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], b = x[4];
    unsigned d0 = x[5], d1 = x[6], d2 = x[7], d3 = x[8], e0 = x[9], e1 = x[10], e2 = x[11], e3 = x[12];
    asm volatile(MMA_F32 "{%0, %1, %2, %3}, {%8, %9}, {%12}, {%0, %1, %2, %3};\n\t"
                 MMA_F32 "{%4, %5, %6, %7}, {%10, %11}, {%12}, {%4, %5, %6, %7};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3), "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b));
    unsigned* y = out + GI * 8;
    y[0] = d0; y[1] = d1; y[2] = d2; y[3] = d3; y[4] = e0; y[5] = e1; y[6] = e2; y[7] = e3;
}

// in: a0 a1, a2 a3, b, d0 d1, e0 e1 (9 words); out: d0 d1, e0 e1.
KERNEL(t_kept_f16) {
    const unsigned* x = in + GI * 9;
    unsigned a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], b = x[4];
    unsigned d0 = x[5], d1 = x[6], e0 = x[7], e1 = x[8];
    asm volatile(MMA_F16 "{%0, %1}, {%4, %5}, {%8}, {%0, %1};\n\t"
                 MMA_F16 "{%2, %3}, {%6, %7}, {%8}, {%2, %3};"
                 : "+r"(d0), "+r"(d1), "+r"(e0), "+r"(e1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b));
    unsigned* y = out + GI * 4;
    y[0] = d0; y[1] = d1; y[2] = e0; y[3] = e1;
}

// in and out as t_kept_f32.
KERNEL(t_kept_again) {
    const unsigned* x = in + GI * 13;
    unsigned a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], b = x[4], t = 0;
    unsigned d0 = x[5], d1 = x[6], d2 = x[7], d3 = x[8], e0 = x[9], e1 = x[10], e2 = x[11], e3 = x[12];
    asm volatile(MMA_F32 "{%0, %1, %2, %3}, {%10, %11}, {%8}, {%0, %1, %2, %3};\n\t"
                 MMA_F32 "{%4, %5, %6, %7}, {%12, %13}, {%8}, {%4, %5, %6, %7};\n\t"
                 "and.b32 %9, %0, 0x80008000;\n\t"
                 "xor.b32 %8, %8, %9;\n\t"
                 MMA_F32 "{%0, %1, %2, %3}, {%12, %13}, {%8}, {%0, %1, %2, %3};\n\t"
                 MMA_F32 "{%4, %5, %6, %7}, {%10, %11}, {%8}, {%4, %5, %6, %7};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3), "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3), "+r"(b), "+r"(t)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3));
    unsigned* y = out + GI * 8;
    y[0] = d0; y[1] = d1; y[2] = d2; y[3] = d3; y[4] = e0; y[5] = e1; y[6] = e2; y[7] = e3;
}

// in: a0 a1, a2 a3, b, d0..d3 (9 words); out: d0..d3.
KERNEL(t_kept_chain) {
    const unsigned* x = in + GI * 9;
    unsigned a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], b = x[4];
    unsigned d0 = x[5], d1 = x[6], d2 = x[7], d3 = x[8];
    asm volatile(MMA_F32 "{%0, %1, %2, %3}, {%4, %5}, {%8}, {%0, %1, %2, %3};\n\t"
                 MMA_F32 "{%0, %1, %2, %3}, {%6, %7}, {%8}, {%0, %1, %2, %3};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b));
    unsigned* y = out + GI * 4;
    y[0] = d0; y[1] = d1; y[2] = d2; y[3] = d3;
}

// in: a0 a1, a2 a3, b (5 words); out: d0..d3, e0..e3, b: two products on one B with nothing to
// add to (the compiler gives the instruction the zero register for its accumulators).
KERNEL(t_kept_zero) {
    const unsigned* x = in + GI * 5;
    unsigned a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], b = x[4];
    unsigned d0 = 0, d1 = 0, d2 = 0, d3 = 0, e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    asm volatile(MMA_F32 "{%0, %1, %2, %3}, {%9, %10}, {%8}, {%0, %1, %2, %3};\n\t"
                 MMA_F32 "{%4, %5, %6, %7}, {%11, %12}, {%8}, {%4, %5, %6, %7};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3), "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3), "+r"(b)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3));
    unsigned* y = out + GI * 9;
    y[0] = d0; y[1] = d1; y[2] = d2; y[3] = d3; y[4] = e0; y[5] = e1; y[6] = e2; y[7] = e3;
    y[8] = b;       // (B is read after the products: the compiler then keeps its register apart from the results.)
}
