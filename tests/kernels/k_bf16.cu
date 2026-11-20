// Kernels that execute bf16 arithmetic instructions which the proxy rewrites for Volta (inline PTX),
// for tcrun.c. Same conventions as k_tensor.cu: every thread reads its operand registers from `in`,
// runs the instructions and writes the result registers to `out`; `aux` is common data.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_90 -O3 -ptx -o k_bf16.135.sm90.ptx k_bf16.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// neg.bf16, add.bf16, mul.bf16, fma.rn.bf16.
// in: a, b, c (bf16 in the low halves). out: neg(a), add(a,b), mul(a,b), fma(a,b,c), in low halves.
// Two kernels with the same body: the harness gives one operands where a rounding decides and the
// other anything.
#define ARITH(name)                                                                                    \
    KERNEL(name) {                                                                                     \
        const unsigned short* x = (const unsigned short*)(in + GI * 3);                                \
        unsigned short a = x[0], b = x[2], c = x[4];                                                   \
        unsigned short d_neg, d_add, d_mul, d_fma;                                                     \
        asm volatile("neg.bf16 %0, %1;" : "=h"(d_neg) : "h"(a));                                       \
        asm volatile("add.bf16 %0, %1, %2;" : "=h"(d_add) : "h"(a), "h"(b));                           \
        asm volatile("mul.bf16 %0, %1, %2;" : "=h"(d_mul) : "h"(a), "h"(b));                           \
        asm volatile("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(d_fma) : "h"(a), "h"(b), "h"(c));            \
        unsigned* y = out + GI * 4;                                                                    \
        y[0] = d_neg; y[1] = d_add; y[2] = d_mul; y[3] = d_fma;                                        \
    }
ARITH(b_arith)
ARITH(b_arith_any)

// add.bf16x2, fma.rn.bf16x2: the same on both halves of a word.
// in: a, b, c (bf16x2). out: add(a,b), fma(a,b,c).
#define ARITH_X2(name)                                                                                 \
    KERNEL(name) {                                                                                     \
        const unsigned* x = in + GI * 3;                                                               \
        unsigned a = x[0], b = x[1], c = x[2];                                                         \
        unsigned d_add, d_fma;                                                                         \
        asm volatile("add.bf16x2 %0, %1, %2;" : "=r"(d_add) : "r"(a), "r"(b));                         \
        asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d_fma) : "r"(a), "r"(b), "r"(c));          \
        out[GI * 2] = d_add;                                                                           \
        out[GI * 2 + 1] = d_fma;                                                                       \
    }
ARITH_X2(b_arith_x2)
ARITH_X2(b_arith_x2_any)

// cvt.rn.bf16.f64: convert an f64 to bf16.
// in: low word of f64, high word of f64. out: bf16 as low 16 bits of a 32-bit word.
KERNEL(b_cvt_f64) {
    const unsigned* x = in + GI * 2;
    unsigned long long dbl = x[0] | ((unsigned long long)x[1] << 32);
    double v;
    __builtin_memcpy(&v, &dbl, 8);
    unsigned short d;
    asm volatile("cvt.rn.bf16.f64 %0, %1;" : "=h"(d) : "d"(v));
    out[GI] = d;
}

// setp.eq.bf16: compare two bf16 for equality.
// in: a (bf16 in low 16), b (bf16 in low 16). out: 1 if equal, 0 otherwise.
KERNEL(b_setp) {
    const unsigned short* x = (const unsigned short*)(in + GI * 2);
    unsigned short a = x[0], b = x[2];
    unsigned result;
    asm volatile("{.reg .pred p;\n\tsetp.eq.bf16 p, %1, %2;\n\tselp.u32 %0, 1, 0, p;}" : "=r"(result) : "h"(a), "h"(b));
    out[GI] = result;
}

// set.eq.bf16x2.bf16x2: compare pairs of bf16, result is bf16x2 (1.0 or 0.0 per element).
// in: a (bf16x2), b (bf16x2). out: set.eq result (bf16x2 as 32-bit word).
KERNEL(b_set_x2) {
    unsigned a = in[GI * 2], b = in[GI * 2 + 1];
    unsigned d;
    asm volatile("set.eq.bf16x2.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    out[GI] = d;
}

// atom.add.noftz.bf16 and atom.add.noftz.bf16x2: every thread of a block adds the same value to one
// word and keeps what the word held before its own sum went in. aux[0] is what the word holds first
// and aux[1] what each thread adds, chosen by the harness so that every partial sum is a bf16 exactly:
// after n sums the word holds aux[0] + n * aux[1], and the values the threads were given back are
// aux[0] + k * aux[1] for k = 0 .. n - 1, each once, in an order nobody knows. Out, the same in every
// thread of a block:
//   [0] for how many k in 0 .. n - 1 exactly one adding thread was given aux[0] + k * aux[1]   (n)
//   [1] what the word holds in the end                                         (aux[0] + n * aux[1])
//   [2] how many of the threads that were not to add still have what their register held   (threads - n)
// The word is in global memory (a word of `out` that is not compared, one per block, through an
// address without a space) or in shared memory (atom.shared). GUARDED: only the threads with an odd
// index add, the instruction being under a predicate that is false for the others.
static __device__ __forceinline__ float wide(unsigned bf16) {
    unsigned bits = bf16 << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}
#define KEPT 0x7fc1  // what the register of a thread that does not add holds, before and after
#define ATOM(name, TYPE, REG, INSN, ADDRESS, SHARED, GUARDED, SAME)                                             \
    KERNEL(name) {                                                                                     \
        __shared__ TYPE in_block, given[64];                                                           \
        __shared__ unsigned char added[64];                                                            \
        TYPE* word = SHARED ? &in_block : (TYPE*)(out + 2048 + blockIdx.x);                            \
        TYPE first = (TYPE)aux[0], each = (TYPE)aux[1], mine;                                          \
        unsigned adds = !(GUARDED) || (threadIdx.x & 1);                                               \
        if (threadIdx.x == 0) *word = first;                                                           \
        __syncthreads();                                                                               \
        /* `word` is an address that says where it points; an instruction with a space wants one */    \
        /* within that space. */                                                                       \
        if (GUARDED)                                                                                   \
            asm volatile("{\n\t.reg .pred p;\n\t.reg .b64 a;\n\t" ADDRESS " a, %1;\n\tsetp.ne.u32 p, %3, 0;\n\t"   \
                         "mov." #REG " %0, %4;\n\t@p " INSN " %0, [a], %2;\n\t}"                       \
                         : "=" SAME(mine) : "l"(word), SAME(each), "r"(adds), SAME((TYPE)KEPT) : "memory"); \
        else                                                                                           \
            asm volatile("{\n\t.reg .b64 a;\n\t" ADDRESS " a, %1;\n\t" INSN " %0, [a], %2;\n\t}"       \
                         : "=" SAME(mine) : "l"(word), SAME(each) : "memory");                          \
        given[threadIdx.x] = mine;                                                                     \
        added[threadIdx.x] = adds;                                                                     \
        __syncthreads();                                                                               \
        unsigned n = 0, once = 0, kept = 0;                                                            \
        for (unsigned i = 0; i < blockDim.x; i++) n += added[i], kept += !added[i] && given[i] == (TYPE)KEPT; \
        for (unsigned k = 0; k < n; k++) {                                                             \
            unsigned seen = 0;                                                                         \
            for (unsigned i = 0; i < blockDim.x; i++) {                                                \
                bool same = added[i];                                                                  \
                for (unsigned half = 0; half < sizeof(TYPE) / 2; half++) {                             \
                    unsigned shift = 16 * half;                                                        \
                    same = same && wide(given[i] >> shift & 0xffff)                                    \
                                   == wide(first >> shift & 0xffff) + (float)k * wide(each >> shift & 0xffff); \
                }                                                                                      \
                seen += same;                                                                          \
            }                                                                                          \
            once += seen == 1;                                                                         \
        }                                                                                              \
        out[GI * 3] = once;                                                                            \
        out[GI * 3 + 1] = *word;                                                                       \
        out[GI * 3 + 2] = kept;                                                                        \
    }
#define H(x) "h"(x)
#define R(x) "r"(x)
ATOM(b_atom, unsigned short, b16, "atom.add.noftz.bf16", "mov.b64", 0, 0, H)
ATOM(b_atom_guarded, unsigned short, b16, "atom.add.noftz.bf16", "mov.b64", 0, 1, H)
ATOM(b_atom_shared, unsigned short, b16, "atom.shared.add.noftz.bf16", "cvta.to.shared.u64", 1, 0, H)
ATOM(b_atom_shared_guarded, unsigned short, b16, "atom.shared.add.noftz.bf16", "cvta.to.shared.u64", 1, 1, H)
ATOM(b_atom_x2, unsigned, b32, "atom.add.noftz.bf16x2", "mov.b64", 0, 0, R)
ATOM(b_atom_x2_guarded, unsigned, b32, "atom.add.noftz.bf16x2", "mov.b64", 0, 1, R)
ATOM(b_atom_x2_shared, unsigned, b32, "atom.shared.add.noftz.bf16x2", "cvta.to.shared.u64", 1, 0, R)
ATOM(b_atom_x2_shared_guarded, unsigned, b32, "atom.shared.add.noftz.bf16x2", "cvta.to.shared.u64", 1, 1, R)
