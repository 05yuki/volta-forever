// Kernels that execute instructions of the GPUs after Ampere which have a plain equivalent on Volta
// (inline PTX), for tcrun.c. Same conventions as k_tensor.cu: every thread reads its operand registers
// from `in`, runs the instructions and writes the result registers to `out`; `aux` is common data.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_100 -O3 -ptx -o k_later.135.sm100.ptx k_later.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// Arithmetic on pairs of floats (sm_100). in: a, b, c, each two f32 in 64 bits; out: a * b + c, a * b to
// nearest and towards minus infinity, a + c, a - c towards zero.
KERNEL(l_f32x2) {
    const unsigned long long* x = (const unsigned long long*)(in + GI * 6);
    unsigned long long d[5];
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d[0]) : "l"(x[0]), "l"(x[1]), "l"(x[2]));
    asm volatile("mul.rn.f32x2 %0, %1, %2;" : "=l"(d[1]) : "l"(x[0]), "l"(x[1]));
    asm volatile("mul.rm.f32x2 %0, %1, %2;" : "=l"(d[2]) : "l"(x[0]), "l"(x[1]));
    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(d[3]) : "l"(x[0]), "l"(x[2]));
    asm volatile("sub.rz.f32x2 %0, %1, %2;" : "=l"(d[4]) : "l"(x[0]), "l"(x[2]));
    unsigned long long* y = (unsigned long long*)(out + GI * 10);
    for (int i = 0; i < 5; i++) y[i] = d[i];
}

// A load and a store with a cache policy (sm_80). in: a byte offset in aux; out: the word there.
KERNEL(l_hint) {
    const unsigned* p = (const unsigned*)((const char*)aux + (in[GI] & ~3u));
    unsigned long long policy;
    unsigned d;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 0.5;" : "=l"(policy));
    asm volatile("ld.global.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(d) : "l"(p), "l"(policy));
    asm volatile("st.global.L2::cache_hint.u32 [%0], %1, %2;" : : "l"(out + GI), "r"(d), "l"(policy) : "memory");
}
