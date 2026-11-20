// tcrun: run the kernels of k_tensor.cu, k_ampere.cu, k_fp8.cu or k_later.cu through the CUDA driver API and check
// every result register against a host model of the instruction (the fragment layouts and the
// definitions of the PTX ISA manual).
//
// usage: tcrun [-d ORDINAL] [-s SEED] [-v] MODULE_FILE
//
// Each kernel runs twice: on values whose sums are exact in half precision, where the results must be
// equal to the model's, and on random values, where they must be within the rounding error of the
// accumulation (or equal, where the model can follow the order of the additions). The kernels that load their operands from shared memory run a second time with rows
// at addresses that are multiples of 4 only. Environment: DRVRUN_LIBCUDA (library to load, default libcuda.so.1), TCRUN_KERNELS
// (comma-separated names, default all found in the module).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
enum { CU_JIT_INFO_LOG_BUFFER = 3, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4, CU_JIT_ERROR_LOG_BUFFER = 5,
       CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6 };

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuDeviceGetName, char*, int, CUdevice);
SYM(CUresult, cuDeviceComputeCapability, int*, int*, CUdevice);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxDestroy_v2, CUcontext);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadDataEx, CUmodule*, const void*, unsigned, int*, void**);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuMemsetD8_v2, CUdeviceptr, unsigned char, size_t);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    void*, void**, void**);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* name) {
    void* p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "missing symbol %s\n", name); exit(2); }
    return p;
}
static const char* ename(CUresult r) {
    const char* s = "?";
    cuGetErrorName(r, &s);
    return s ? s : "?";
}
#define CK(call) do { CUresult r_ = (call); if (r_) { printf("FAIL %s -> %d %s\n", #call, r_, ename(r_)); exit(1); } } while (0)

// Geometry: GRID blocks of two warps; the matrices of every warp differ.
#define BLOCK 64
#define GRID 3
#define THREADS (GRID * BLOCK)
#define WARPS (THREADS / 32)
#define MAXREG 48
#define AUX_WORDS 1024
#define LOOP_STEPS 5

typedef uint32_t reg[32];  // one register in the 32 lanes of a warp

// ---- values -----------------------------------------------------------------------------------------
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static int rnd_in(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

static double h2d(uint32_t h) {  // IEEE half -> double
    int sign = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? ldexp(m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexp(m + 1024, e - 25);
    return sign ? -v : v;
}
static uint32_t d2h(double v) {  // double -> IEEE half, round to nearest even
    uint32_t sign = signbit(v) ? 0x8000 : 0;
    v = fabs(v);
    if (isnan(v)) return 0x7e00;
    if (v >= 65520.0) return sign | 0x7c00;
    if (v < ldexp(1, -14)) return sign | (uint32_t)nearbyint(ldexp(v, 24));  // subnormal or zero
    int e;
    double m = frexp(v, &e);                              // v = m * 2^e, 0.5 <= m < 1
    uint32_t bits = (uint32_t)nearbyint(ldexp(m, 11));    // 1024..2048
    if (bits == 2048) { bits = 1024; e++; }
    return sign | ((uint32_t)(e + 14) << 10) | (bits - 1024);
}
static double f2d(uint32_t bits) { float f; memcpy(&f, &bits, 4); return f; }
static uint32_t d2f(double v) { float f = (float)v; uint32_t bits; memcpy(&bits, &f, 4); return bits; }
static double lo(uint32_t r) { return h2d(r & 0xffff); }
static double hi(uint32_t r) { return h2d(r >> 16); }
static uint32_t pack(double l, double h) { return d2h(l) | d2h(h) << 16; }

// ---- FP8 by the definition of the formats ---------------------------------------------------------------
// e4m3: sign, 4 bits of exponent (bias 7), 3 of mantissa; no infinity, 0x7f and 0xff are the NaN; the
// largest value is 448 (0x7e). e5m2: sign, 5 bits of exponent (bias 15), 2 of mantissa, infinities and
// NaN as in a half; the largest finite value is 57344 (0x7b).
static double fp8_value(int e5m2, uint8_t byte) {
    int sign = byte >> 7, magnitude = byte & 0x7f;
    double v;
    if (e5m2) {
        int e = magnitude >> 2, m = magnitude & 3;
        v = e == 31 ? (m ? NAN : INFINITY) : e == 0 ? ldexp(m, -16) : ldexp(4 + m, e - 17);
    } else {
        int e = magnitude >> 3, m = magnitude & 7;
        v = magnitude == 0x7f ? NAN : e == 0 ? ldexp(m, -9) : ldexp(8 + m, e - 10);
    }
    return sign ? -v : v;
}
static int fp8_top(int e5m2) { return e5m2 ? 0x7b : 0x7e; }
static int fp8_is_nan(int e5m2, uint8_t byte) {
    return e5m2 ? (byte & 0x7c) == 0x7c && (byte & 3) : (byte & 0x7f) == 0x7f;
}
// The byte of the format nearest to v, ties to the even byte; the largest finite one beyond it.
static uint8_t fp8_nearest(int e5m2, double v) {
    uint8_t sign = signbit(v) ? 0x80 : 0;
    if (isnan(v)) return 0x7f;
    double a = fabs(v);
    int top = fp8_top(e5m2), best = 0;
    if (a >= fp8_value(e5m2, top)) return sign | top;
    for (int code = 1; code <= top; code++) {
        double here = fabs(a - fp8_value(e5m2, code)), there = fabs(a - fp8_value(e5m2, best));
        if (here < there || (here == there && (code & 1) == 0)) best = code;
    }
    return sign | best;
}
// An f32 at a value of the format, next to one, or halfway between two (where the rounding decides).
static uint32_t fp8_critical(int e5m2) {
    int top = fp8_top(e5m2), code = rnd() % (top + 1);
    double v = fp8_value(e5m2, code), next = code < top ? fp8_value(e5m2, code + 1) : v * 1.0625;
    float middle = (float)((v + next) / 2), f;
    switch (rnd() % 7) {
    case 0: f = (float)v; break;
    case 1: f = middle; break;
    case 2: f = nextafterf(middle, 0); break;
    case 3: f = nextafterf(middle, INFINITY); break;
    case 4: f = nextafterf((float)v, INFINITY); break;
    case 5: f = nextafterf((float)next, 0); break;
    default: f = (float)(v + (next - v) * (rnd() % 1024) / 1024.0); break;
    }
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return bits ^ (rnd() & 0x80000000u);
}
// One FP8 byte. set 0: (exact) 0, +-0.5, +-1, +-1.5, +-2, +-3, +-4; (random) a magnitude from 1/8 to 8,
// or one time in 64 a NaN or an infinity (the sums of such magnitudes stay within the tolerance of P;
// a large value among them would not). set 1: one of the 16 smallest magnitudes. set 2: 0, +-1, +-2.
// set 3: one of the 8 largest magnitudes. The products of a byte of set 1 or 3 and one of set 2 add up
// exactly, on accumulators that are multiples of 1/64 (set 1) or whole (set 3).
static uint8_t fp8_byte(int e5m2, int set, int mode) {
    static const double exact[] = {0, 0.5, 1, 1.5, 2, 3, 4}, whole[] = {0, 1, 2};
    uint8_t sign = rnd() & 0x80;
    if (set == 1) return sign | rnd() % 16;
    if (set == 2) return sign | fp8_nearest(e5m2, whole[rnd() % 3]);
    if (set == 3) return sign | (fp8_top(e5m2) - rnd() % 8);
    if (mode == 0) return sign | fp8_nearest(e5m2, exact[rnd() % 7]);
    if (rnd() % 64 == 0) return sign | (e5m2 ? 0x7c + rnd() % 4 : 0x7f);
    // The exponent fields of 1/8 to 4, and any mantissa.
    return sign | (e5m2 ? rnd_in(12 << 2, (18 << 2) - 1) : rnd_in(4 << 3, (10 << 3) - 1));
}

// ---- E2M1 and UE8M0 data functions (used by the value generators and the model functions) ----------

// E2M1 value table: code 0..15 -> double. 1 sign, 2 exponent (bias 1), 1 mantissa. No inf, no NaN.
static double e2m1_value(uint8_t code) {
    static const double table[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};
    return table[code & 0xf];
}

// The E2M1 nibble nearest to `v`, ties to even (even code has m0=0). Beyond +-6 -> +-6; NaN -> +6.
static uint8_t e2m1_nearest(double v) {
    if (isnan(v)) return 7;
    uint8_t sign = signbit(v) ? 8 : 0;
    double a = fabs(v);
    if (a >= 6.0) return sign | 7;
    uint8_t best = 0;
    double best_d = a;
    for (uint8_t c = 1; c < 8; c++) {
        double d = fabs(a - e2m1_value(c));
        if (d < best_d || (d == best_d && (c & 1) == 0)) { best = c; best_d = d; }
    }
    return sign | best;
}

// An f32 value that tests E2M1 conversion boundaries.
static uint32_t e2m1_critical(void) {
    static const double boundaries[] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0};
    static const double values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    double v;
    int choice = rnd() % 10;
    if (choice < 4) {
        v = boundaries[rnd() % 7];
        uint32_t bits = d2f(v);
        bits += (int)(rnd() % 3) - 1;
        v = f2d(bits);
    } else if (choice < 7) {
        v = values[rnd() % 8];
    } else if (choice == 7) {
        v = 6.0 + (rnd() % 100);
    } else if (choice == 8) {
        static const uint32_t special[] = {0x7f800000, 0xff800000, 0x7fc00000, 0, 0x80000000};
        return special[rnd() % 5];
    } else {
        v = ldexp(1.0 + (rnd() & 0x7fffff) / 8388608.0, rnd_in(-2, 3));
    }
    if (rnd() & 1) v = -v;
    return d2f(v);
}

// The UE8M0 byte for |v| with the given rounding (0=rn, 1=rz, 2=rp). satfinite: inf->0xfe, NaN->0xff.
// The representable set is {2^(-127), 2^(-126), ..., 2^(127), NaN}, where byte 0 = 2^(-127).
// Verified against RTX 5090 hardware.
static uint8_t ue8m0_round(double v, int rounding) {
    if (isnan(v)) return 0xff;
    double a = fabs(v);
    if (a == 0) return 0;
    if (isinf(a)) return 0xfe;
    int e;
    double m = frexp(a, &e);  // a = m * 2^e, 0.5 <= m < 1
    int actual_exp = e - 1;    // a in [2^actual_exp, 2^(actual_exp+1))
    int is_exact = (m == 0.5); // a == 2^actual_exp exactly
    int floor_byte = actual_exp + 127;
    int ceil_byte = is_exact ? floor_byte : floor_byte + 1;
    // Clamp to [0, 254]
    if (floor_byte < 0) floor_byte = 0;
    if (ceil_byte < 0) ceil_byte = 0;
    if (floor_byte > 254) floor_byte = 254;
    if (ceil_byte > 254) ceil_byte = 254;
    switch (rounding) {
    case 0: { // rn: round to nearest, ties to even
        if (is_exact) return floor_byte;
        // m is the fraction: m in (0.5, 1). Distance from 2^actual_exp is proportional to m - 0.5.
        double frac = m * 2 - 1;  // how far past the floor power (0 = at floor, 1 = at ceil)
        if (frac > 0.5) return ceil_byte;
        if (frac < 0.5) return floor_byte;
        // Tie: round to even byte
        return (floor_byte & 1) == 0 ? floor_byte : ceil_byte;
    }
    case 1: return floor_byte;
    case 2: return ceil_byte;
    }
    return 0;
}

// An f32 value that tests UE8M0 conversion boundaries. rounding: 1=rz, 2=rp.
// Includes the subnormal range near 2^(-127) where the hardware's behavior was initially unknown.
static uint32_t ue8m0_critical(int rounding) {
    (void)rounding;
    int n = rnd_in(-127, 126);  // include -127 (the byte 0 boundary)
    double base = ldexp(1.0, n), v;
    int choice = rnd() % 10;
    if (choice < 3) {
        v = base * 1.5;
        uint32_t bits = d2f(v);
        bits += (int)(rnd() % 3) - 1;
        v = f2d(bits);
    } else if (choice < 5) {
        v = base;
    } else if (choice == 5) {
        v = rnd() & 1 ? ldexp(1.0, -126) : ldexp(1.0, 127);
        uint32_t bits = d2f(v);
        bits += (int)(rnd() % 3) - 1;
        v = f2d(bits);
    } else if (choice == 6) {
        static const uint32_t special[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000};
        return special[rnd() % 5];
    } else if (choice == 7) {
        // f32 subnormals near 2^(-127) = 0x00400000: the boundary between byte 0 and byte 1
        uint32_t bits = 0x00400000 + (int)(rnd() % 5) - 2;
        if (rnd() & 1) bits |= 0x80000000;  // negative
        return bits;
    } else if (choice == 8) {
        // very small subnormals
        return rnd() % 20 + 1;
    } else {
        v = ldexp(1.0 + (rnd() & 0x7fffff) / 8388608.0, rnd_in(-126, 127));
    }
    if (rnd() & 1) v = -v;
    return d2f(v);
}

// One input register of a given kind. mode 0: exact sums; mode 1: random.
//   E Q J G: four e4m3 bytes of set 0, 1, 2, 3 (see fp8_byte)   M R K W: four e5m2 bytes likewise
//   I: an f32 accumulator that is a whole number, in both modes
//   e: two FP8 bytes of any value   x, y: an f32 where the conversion to e4m3 (x) or e5m2 (y) rounds
//   B: two bf16 elements   T: a tf32 (mode 1: one time in four an f32 that is not rounded to one)
//   L, H: the low and the high word of a double (H: of the double that the last L began)
//   A: an f32 accumulator that is a multiple of 1/64, in both modes
//   h: two f16 multiplicand elements   c: two f16 accumulator elements   F: f32 accumulator
//   w: any 32 bits (four int8, an s32 accumulator)   o: byte offset of a 16-byte row in shared memory
//   S: an s32 accumulator within a sum of products of either end of its range, or near zero
//   u: such an offset that is a multiple of 4 only   n: the trip count of a loop
//   z: how many of 16 bytes a copy takes (all or nothing in mode 0)   g: an f32 of any size (in mode 0,
//   one a half holds exactly)   N: such an f32, or a NaN, an infinity or a zero
// ---- bf16: the upper half of an f32 (a sign, eight bits of exponent, seven of mantissa) -------------
static double bf16_value(uint32_t bits) { return f2d((bits & 0xffff) << 16); }
// The magnitude of the bf16 whose bits without the sign are `pattern` (0 to 0x7f80: they are in the
// order of their values), with 2^128 where the pattern is infinity, which is what the values beyond
// the largest finite one are compared with when they are rounded.
static double bf16_magnitude(uint32_t pattern) { return pattern == 0x7f80 ? 0x1p128 : bf16_value(pattern); }
static uint32_t bf16_any(void) {
    static const uint32_t special[] = {0x7f80, 0xff80, 0x7fc0, 0, 0x8000, 0x0001, 0x807f};
    if (rnd() % 16 == 0) return special[rnd() % 7];
    // Exponents near one another (sums that mix their bits) or anywhere.
    uint32_t exponent = rnd() & 1 ? rnd_in(115, 139) : rnd_in(0, 254);
    return (rnd() & 0x8000) | exponent << 7 | (rnd() & 0x7f);
}
// The operands of one a * b + c. mode 0: those where a rounding decides. Mostly, the product of the
// two mantissas is a halfway point of bf16 (eight bits, a one, then zeros) and c is nothing, or
// far too small for an f32 to hold beside the product, or just large enough, on one side or the other:
// the result is the even neighbour, the one above or the one below, and an f32 sum rounded to nearest
// would be the halfway point itself. Otherwise small whole numbers, sums that cancel, and products in
// the subnormal range. mode 1: anything.
static uint32_t bf16_abc[2][3];
static void bf16_operands(int mode, uint32_t* abc) {
    static uint8_t halfway[16384][2];
    static int halfways = -1;
    if (halfways < 0) {
        halfways = 0;
        for (int a = 128; a < 256; a++) for (int b = 128; b < 256; b++) {
            int product = a * b, length = product >> 15 ? 16 : 15;
            if ((product & ((1 << (length - 8)) - 1)) == 1 << (length - 9)) halfway[halfways][0] = a, halfway[halfways++][1] = b;
        }
    }
    if (mode == 1) { for (int i = 0; i < 3; i++) abc[i] = bf16_any(); return; }
    int choice = rnd() % 8;
    if (choice < 5) {
        static const int below[] = {9, 10, 16, 17, 23, 24, 25, 26, 40, 60, 80};
        const uint8_t* m = halfway[rnd() % halfways];
        int ea = rnd_in(-20, 20), eb = rnd_in(-20, 20);
        double a = ldexp(m[0], ea - 7), b = ldexp(m[1], eb - 7), c = choice == 0 ? 0 : ldexp(1, ea + eb - below[rnd() % 11]);
        if (rnd() & 1) a = -a;
        if (rnd() & 1) b = -b;
        if (rnd() & 1) c = -c;
        abc[0] = d2f(a) >> 16, abc[1] = d2f(b) >> 16, abc[2] = d2f(c) >> 16;
    } else if (choice == 5) {
        for (int i = 0; i < 3; i++) abc[i] = d2f(rnd_in(-8, 8) * 0.5) >> 16;
    } else if (choice == 6) {
        int a = rnd_in(-15, 15), b = rnd_in(-15, 15);
        abc[0] = d2f(a) >> 16, abc[1] = d2f(b) >> 16, abc[2] = d2f(-a * b + rnd_in(-1, 1) * (int)(rnd() & 1)) >> 16;
    } else {
        abc[0] = d2f(ldexp(rnd_in(-255, 255), -70)) >> 16, abc[1] = d2f(ldexp(rnd_in(-255, 255), -66 - (int)(rnd() % 6))) >> 16;
        abc[2] = d2f(ldexp(rnd_in(-3, 3), -133)) >> 16;
    }
}
// See value 'a'.
static uint32_t bf16_sum_operand(int mode, int element) {
    static const int exact[] = {-133, -130, -126, -1, 0, 7, 119};
    static int calls[2], exponent[2];
    if (calls[element]++ % 2) return d2f(ldexp(rnd() & 1 ? 1 : -1, exponent[element])) >> 16;
    exponent[element] = mode == 0 ? exact[rnd() % 7] : rnd_in(-133, 119);
    return d2f(ldexp(rnd_in(-191, 191), exponent[element])) >> 16;
}

static uint32_t value(char kind, int mode) {
    static uint64_t wide;
    switch (kind) {
    case 'E': case 'Q': case 'J': case 'G': case 'M': case 'R': case 'K': case 'W': {
        int e5m2 = kind == 'M' || kind == 'R' || kind == 'K' || kind == 'W';
        int set = kind == 'E' || kind == 'M' ? 0 : kind == 'Q' || kind == 'R' ? 1 : kind == 'J' || kind == 'K' ? 2 : 3;
        uint32_t bytes = 0;
        for (int i = 0; i < 4; i++) bytes |= (uint32_t)fp8_byte(e5m2, set, mode) << (8 * i);
        return bytes;
    }
    case 'e': return mode == 0 ? (rnd() % 11 | (rnd() & 0x80)) | (rnd() % 11 | (rnd() & 0x80)) << 8 : rnd() & 0xffff;
    case 'x': return fp8_critical(0);
    case 'y': return fp8_critical(1);
    case 'B': {
        static const uint32_t special[] = {0x7f80, 0xff80, 0x7fc0, 0, 0x8000, 0x0001, 0x807f};
        uint32_t pair = 0;
        for (int i = 0; i < 2; i++) {
            uint32_t element = mode == 0 ? d2f(rnd_in(-8, 8) * 0.5) >> 16
                             : rnd() % 64 == 0 ? special[rnd() % 7]
                             : (rnd() & 0x8000) | (uint32_t)rnd_in(115, 139) << 7 | (rnd() & 0x7f);
            pair |= element << (16 * i);
        }
        return pair;
    }
    case 'T': {
        static const uint32_t special[] = {0x7f800000, 0xff800000, 0x7fc00000, 0, 0x80000000, 0x00002000, 0x80400000};
        if (mode == 0) return d2f(rnd_in(-8, 8) * 0.5);
        if (rnd() % 64 == 0) return special[rnd() % 7];
        uint32_t mantissa = rnd() % 4 == 0 ? rnd() & 0x7fffff : (rnd() & 0x3ff) << 13;
        return (rnd() & 0x80000000u) | (uint32_t)rnd_in(115, 139) << 23 | mantissa;
    }
    case 'L': {
        double v = mode == 0 ? rnd_in(-8, 8) * 0.5
                             : ldexp(1.0 + (((uint64_t)rnd() << 20 ^ rnd()) & 0xfffffffffffffull) / 4503599627370496.0, rnd_in(-20, 20));
        if (mode && (rnd() & 1)) v = -v;
        memcpy(&wide, &v, 8);
        return (uint32_t)wide;
    }
    case 'H': return wide >> 32;
    case 'A': return d2f(rnd_in(-4096, 4096) / 64.0);
    case 'I': return d2f(rnd_in(-64, 64));
    case 'z': { uint32_t r = rnd(); return mode == 0 ? (r & 1 ? 16 : 0) : r % 17; }
    case 'g': {
        if (mode == 0) return d2f(rnd_in(-1024, 1024) * 0.03125);
        double v = ldexp(1.0 + (rnd() & 0x7fffff) / 8388608.0, rnd_in(-20, 17));  // overflows and underflows a half
        return d2f(rnd() & 1 ? -v : v);
    }
    case 'N': {
        static const uint32_t special[] = {0x7fc00000, 0xffc00001, 0x7f800001, 0, 0x80000000, 0x7f800000, 0xff800000};
        uint32_t r = rnd() % 14;
        return r < 7 ? special[r] : value('g', 1);
    }
    case 'h': return mode == 0 ? pack(rnd_in(-4, 4) * 0.5, rnd_in(-4, 4) * 0.5)
                               : pack(rnd_in(-1024, 1024) / 1024.0, rnd_in(-1024, 1024) / 1024.0);
    case 'c': return mode == 0 ? pack(rnd_in(-32, 32) * 0.25, rnd_in(-32, 32) * 0.25)
                               : pack(rnd_in(-512, 512) / 128.0, rnd_in(-512, 512) / 128.0);
    case 'F': return mode == 0 ? d2f(rnd_in(-32, 32) * 0.25) : d2f(rnd_in(-1000000, 1000000) / 250000.0);
    case 'o': return 16 * (rnd() % (AUX_WORDS * 4 / 16));
    case 'u': return 4 * (rnd() % (AUX_WORDS - 3));
    case 'n': return LOOP_STEPS;
    case 'S': {
        // 32 products of two int8 reach 2^19: within that of the largest and of the smallest s32,
        // a saturating accumulation saturates for some sums and not for others.
        int32_t near = rnd_in(0, 1 << 19);
        switch (rnd() % 3) {
        case 0: return (uint32_t)(INT32_MAX - near);
        case 1: return (uint32_t)(INT32_MIN + near);
        default: return (uint32_t)(near - (1 << 18));
        }
    }
    // m: a bf16 in the low half of a word. mode 0: small multiples of 1/2 (equal ones are common);
    // mode 1: anything, zeros, infinities, NaN and subnormals among them.
    case 'm': return mode == 0 ? d2f(rnd_in(-8, 8) * 0.5) >> 16 : bf16_any();
    // d, p, q: the operands a, b, c of a * b + c, as bf16 in the low halves (d makes the three).
    // r, s, t: the same in both halves of a word, two products.
    case 'd': bf16_operands(mode, bf16_abc[0]); return bf16_abc[0][0];
    case 'p': return bf16_abc[0][1];
    case 'q': return bf16_abc[0][2];
    case 'r': bf16_operands(mode, bf16_abc[0]); bf16_operands(mode, bf16_abc[1]); return bf16_abc[0][0] | bf16_abc[1][0] << 16;
    case 's': return bf16_abc[0][1] | bf16_abc[1][1] << 16;
    case 't': return bf16_abc[0][2] | bf16_abc[1][2] << 16;
    // D (with H for its upper word): an f64 to convert to bf16. mode 0: where the rounding decides,
    // a bf16, the halfway point to the next one, and the doubles beside both, as near as a double can
    // be and at the distances an f32 would or would not tell from them; the largest values, the
    // smallest. mode 1: anything an f32 could be near, and what it could not.
    case 'D': {
        double v;
        uint32_t pattern = rnd() % 16 == 0 ? 0x7f7f - rnd() % 2 : rnd() % 16 == 1 ? rnd() % 3 : rnd() % 0x7f80;
        if (mode == 0) {
            static const int distance[] = {20, 23, 24, 25, 30, 45};
            v = rnd() & 1 ? bf16_magnitude(pattern) : (bf16_magnitude(pattern) + bf16_magnitude(pattern + 1)) / 2;
            switch (rnd() % 5) {
            case 0: break;
            case 1: v = nextafter(v, INFINITY); break;
            case 2: v = nextafter(v, 0); break;
            case 3: v += ldexp(v, -distance[rnd() % 6]); break;
            default: v -= ldexp(v, -distance[rnd() % 6]); break;
            }
        } else {
            static const double special[] = {NAN, INFINITY, 0.0, 1e300, 1e-300, 0x1p-134, 0x1p-135, 0x1p128};
            v = rnd() % 16 == 0 ? special[rnd() % 8]
              : ldexp(1.0 + rnd() / 4294967296.0 + rnd() / 18446744073709551616.0, rnd_in(-140, 130));
        }
        if (rnd() & 1) v = -v;
        memcpy(&wide, &v, 8);
        return (uint32_t)wide;
    }
    // a: what a word holds first, then what every thread adds to it (aux words 0 and 1, and so on
    // alternately): m * 2^e and 2^e, with m small enough for every partial sum of a block to be a
    // bf16 exactly, in whatever order the threads add. v: the same in both halves.
    case 'a': return bf16_sum_operand(mode, 0);
    case 'v': { uint32_t low = bf16_sum_operand(mode, 0); return low | bf16_sum_operand(mode, 1) << 16; }
    // f: an E2M1 byte (two nibbles) for decode testing. mode 0: all 16 values; mode 1: random byte.
    case 'f': return mode == 0 ? (rnd() % 16) | (rnd() % 16) << 4 : rnd() & 0xff;
    // 1: an f32 at a rounding boundary of E2M1.   2: same for UE8M0 rp.   3: UE8M0 rz.
    case '1': return e2m1_critical();
    case '2': return ue8m0_critical(2);
    case '3': return ue8m0_critical(1);
    // U: a UE8M0 pair (two bytes) for decode testing. mode 0: all 256 values; mode 1: random.
    case 'U': return mode == 0 ? (rnd() % 256) | (rnd() % 256) << 8 : rnd() & 0xffff;
    default: return rnd();
    }
}

// ---- models: the instructions on the registers of one warp --------------------------------------------
// Volta mma.m8n8k4: four 8x8x4 products, one per octet.
static void ref_v884(int a_col, int b_col, int d_f32, int c_f32, reg* d, reg* a, reg* b, reg* c) {
    for (int q = 0; q < 4; q++) {
        double A[8][4], B[4][8], C[8][8], D[8][8];
        for (int h = 0; h < 2; h++) for (int k = 0; k < 4; k++) {
            int lane = 16 * h + 4 * q + k;
            for (int i = 0; i < 4; i++) {
                double av = i & 1 ? hi(a[i / 2][lane]) : lo(a[i / 2][lane]);
                double bv = i & 1 ? hi(b[i / 2][lane]) : lo(b[i / 2][lane]);
                if (a_col) A[i + 4 * h][k] = av; else A[k + 4 * h][i] = av;
                if (b_col) B[i][k + 4 * h] = bv; else B[k][i + 4 * h] = bv;
            }
            for (int i = 0; i < 8; i++) {
                if (c_f32) C[(k & 1) + (i & 2) + 4 * h][(i & 4) + (k & 2) + (i & 1)] = f2d(c[i][lane]);
                else C[k + 4 * h][i] = i & 1 ? hi(c[i / 2][lane]) : lo(c[i / 2][lane]);
            }
        }
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
            D[i][j] = C[i][j];
            for (int k = 0; k < 4; k++) D[i][j] += A[i][k] * B[k][j];
        }
        for (int h = 0; h < 2; h++) for (int k = 0; k < 4; k++) {
            int lane = 16 * h + 4 * q + k;
            if (d_f32) for (int i = 0; i < 8; i++) d[i][lane] = d2f(D[(k & 1) + (i & 2) + 4 * h][(i & 4) + (k & 2) + (i & 1)]);
            else for (int i = 0; i < 4; i++) d[i][lane] = pack(D[k + 4 * h][2 * i], D[k + 4 * h][2 * i + 1]);
        }
    }
}
// Turing mma.m16n8k8 with f32 (4 registers) or f16 (2 registers) accumulators.
static void ref_mma16(int f16, reg* d, reg* a, uint32_t* b, reg* c) {
    double A[16][8], B[8][8], C[16][8], D[16][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        A[g][2 * t] = lo(a[0][lane]); A[g][2 * t + 1] = hi(a[0][lane]);
        A[g + 8][2 * t] = lo(a[1][lane]); A[g + 8][2 * t + 1] = hi(a[1][lane]);
        B[2 * t][g] = lo(b[lane]); B[2 * t + 1][g] = hi(b[lane]);
        for (int i = 0; i < 4; i++) {
            double v = f16 ? (i & 1 ? hi(c[i / 2][lane]) : lo(c[i / 2][lane])) : f2d(c[i][lane]);
            C[g + 8 * (i / 2)][2 * t + (i & 1)] = v;
        }
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++) {
        D[i][j] = C[i][j];
        for (int k = 0; k < 8; k++) D[i][j] += A[i][k] * B[k][j];
    }
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        if (f16) for (int i = 0; i < 2; i++) d[i][lane] = pack(D[g + 8 * i][2 * t], D[g + 8 * i][2 * t + 1]);
        else for (int i = 0; i < 4; i++) d[i][lane] = d2f(D[g + 8 * (i / 2)][2 * t + (i & 1)]);
    }
}
// Ampere mma.m16n8k16 (FP16) with f32 (4 registers) or f16 (2 registers) accumulators: A is 16x16 in
// four registers, B 16x8 in two.
static void ref_mma16k16(int f16, reg* d, reg* a, reg* b, reg* c) {
    double A[16][16], B[16][8], C[16][8], D[16][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 8; i++) {  // element i: register i / 2, its upper half if i is odd
            double v = i & 1 ? hi(a[i / 2][lane]) : lo(a[i / 2][lane]);
            A[(i < 2 || (i >= 4 && i < 6)) ? g : g + 8][2 * t + (i & 1) + (i >= 4 ? 8 : 0)] = v;
        }
        for (int i = 0; i < 4; i++) {
            double v = i & 1 ? hi(b[i / 2][lane]) : lo(b[i / 2][lane]);
            B[2 * t + (i & 1) + (i >= 2 ? 8 : 0)][g] = v;
        }
        for (int i = 0; i < 4; i++) {
            double v = f16 ? (i & 1 ? hi(c[i / 2][lane]) : lo(c[i / 2][lane])) : f2d(c[i][lane]);
            C[g + 8 * (i / 2)][2 * t + (i & 1)] = v;
        }
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++) {
        D[i][j] = C[i][j];
        for (int k = 0; k < 16; k++) D[i][j] += A[i][k] * B[k][j];
    }
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        if (f16) for (int i = 0; i < 2; i++) d[i][lane] = pack(D[g + 8 * i][2 * t], D[g + 8 * i][2 * t + 1]);
        else for (int i = 0; i < 4; i++) d[i][lane] = d2f(D[g + 8 * (i / 2)][2 * t + (i & 1)]);
    }
}

// Turing mma.m8n8k16 on bytes: signed_a, signed_b tell how the elements are read.
static void ref_mma8(int signed_a, int signed_b, reg* d, uint32_t* a, uint32_t* b, reg* c) {
    int32_t A[8][16], B[16][8];
    uint32_t D[8][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) {
            uint8_t ab = a[lane] >> (8 * i), bb = b[lane] >> (8 * i);
            A[g][4 * t + i] = signed_a ? (int8_t)ab : ab;
            B[4 * t + i][g] = signed_b ? (int8_t)bb : bb;
        }
        D[g][2 * t] = c[0][lane]; D[g][2 * t + 1] = c[1][lane];
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < 16; k++) D[i][j] += (uint32_t)(A[i][k] * B[k][j]);
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        d[0][lane] = D[g][2 * t]; d[1][lane] = D[g][2 * t + 1];
    }
}
// movmatrix.trans: the 8x8 matrix of 16-bit elements held one row per group, transposed.
static void ref_movmatrix(uint32_t* d, uint32_t* a) {
    uint16_t M[8][8];
    for (int lane = 0; lane < 32; lane++) {
        M[lane >> 2][2 * (lane & 3)] = a[lane] & 0xffff;
        M[lane >> 2][2 * (lane & 3) + 1] = a[lane] >> 16;
    }
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        d[lane] = M[2 * t][g] | (uint32_t)M[2 * t + 1][g] << 16;
    }
}
// ldmatrix: lanes 8m..8m+7 give the addresses of the rows of matrix m.
static uint16_t shared16(const uint32_t* aux, int block, uint32_t byte) {
    uint32_t word = aux[byte / 4] + (uint32_t)block * 0x00010001u;
    return byte & 2 ? word >> 16 : word & 0xffff;
}
static void ref_ldmatrix(int count, int trans, reg* r, uint32_t* offset, const uint32_t* aux, int block) {
    for (int m = 0; m < count; m++) for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        uint16_t e0, e1;
        if (trans) {
            e0 = shared16(aux, block, offset[8 * m + 2 * t] + 2 * g);
            e1 = shared16(aux, block, offset[8 * m + 2 * t + 1] + 2 * g);
        } else {
            e0 = shared16(aux, block, offset[8 * m + g] + 4 * t);
            e1 = shared16(aux, block, offset[8 * m + g] + 4 * t + 2);
        }
        r[m][lane] = e0 | (uint32_t)e1 << 16;
    }
}

// ---- the kernels ----------------------------------------------------------------------------------------
typedef void model_fn(int variant, reg* out, reg* in, const uint32_t* aux, int block);

static void m_v884(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    // variant: bit 0 = B column-major, bit 1 = A column-major, bits 2-3 = 0 f32/f32, 1 f16/f16, 2 f32 d / f16 c
    (void)aux; (void)block;
    ref_v884(v >> 1 & 1, v & 1, (v >> 2) != 1, (v >> 2) == 0, out, in, in + 2, in + 4);
}
static void m_mma16(int f16, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    ref_mma16(f16, out, in, in[2], in + 3);
}
static void m_mma8(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    ref_mma8(v & 2, v & 1, out, in[0], in[1], in + 2);
}
static void m_movmatrix(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_movmatrix(out[0], in[0]);
}
static void m_ldmatrix(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    ref_ldmatrix(v & 7, v & 8, out, in[0], aux, block);  // variant: count, +8 for trans
}
static void m_chain_f(int f16, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    reg* a = in; reg* b = in + 4; int n = f16 ? 2 : 4;
    memcpy(out, in + 8, 2 * n * sizeof(reg));
    ref_mma16(f16, out, a, b[0], out);
    ref_mma16(f16, out, a + 2, b[2], out);
    ref_mma16(f16, out + n, a, b[1], out + n);
    ref_mma16(f16, out + n, a + 2, b[3], out + n);
}
static void m_chain_s8(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    memcpy(out, in + 6, 4 * sizeof(reg));
    ref_mma8(1, 1, out, in[0], in[4], out);
    ref_mma8(1, 1, out + 2, in[1], in[4], out + 2);
    ref_mma8(1, 1, out, in[2], in[5], out);
    ref_mma8(1, 1, out + 2, in[3], in[5], out + 2);
}
// k_chains.cu: chain whose accumulator is read by a non-product statement between groups.
// in: a0 a1 (h) b0 b1 b2 b3 (h) c0..c3 (F) scalar (F). out: d0..d3 (F).
static void m_chain_end_read(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    reg a[2]; memcpy(a, in, 2 * sizeof(reg));
    reg b[4]; memcpy(b, in + 2, 4 * sizeof(reg));
    memcpy(out, in + 6, 4 * sizeof(reg));
    // First chain of 4 products
    ref_mma16(0, out, a, b[0], out);
    ref_mma16(0, out, a, b[1], out);
    ref_mma16(0, out, a, b[2], out);
    ref_mma16(0, out, a, b[3], out);
    // Non-product read: add scalar to each accumulator register (per lane)
    for (int lane = 0; lane < 32; lane++) {
        float scalar; memcpy(&scalar, &in[10][lane], 4);
        for (int i = 0; i < 4; i++) {
            float fv; memcpy(&fv, &out[i][lane], 4);
            fv += scalar;
            memcpy(&out[i][lane], &fv, 4);
        }
    }
    // Second chain of 4 products
    ref_mma16(0, out, a, b[0], out);
    ref_mma16(0, out, a, b[1], out);
    ref_mma16(0, out, a, b[2], out);
    ref_mma16(0, out, a, b[3], out);
}
// t_chain_carry: two chains with a non-touching statement between them (carry test).
// in: a0 a1 (h) b0 b1 b2 b3 (h) c0..c3 (F) extra (F). out: d0..d3 (F).
static void m_chain_carry(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    reg a[2]; memcpy(a, in, 2 * sizeof(reg));
    reg b[4]; memcpy(b, in + 2, 4 * sizeof(reg));
    memcpy(out, in + 6, 4 * sizeof(reg));
    for (int r = 0; r < 2; r++) {
        ref_mma16(0, out, a, b[0], out);
        ref_mma16(0, out, a, b[1], out);
        ref_mma16(0, out, a, b[2], out);
        ref_mma16(0, out, a, b[3], out);
    }
}
// t_chain_back: aux[0] turns of a chain of 4 products on one accumulator (a loop with its back
// edge), then one product on another.
// in: a0 a1 (h) b0 b1 b2 b3 (h) c0..c3 (F) e0..e3 (F). out: d0..d3 e0..e3 (F).
#define CHAIN_TURNS 5
static void m_chain_back(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[2]; memcpy(a, in, 2 * sizeof(reg));
    reg b[4]; memcpy(b, in + 2, 4 * sizeof(reg));
    memcpy(out, in + 6, 4 * sizeof(reg));
    memcpy(out + 4, in + 10, 4 * sizeof(reg));
    for (uint32_t turn = 0; turn < aux[0]; turn++) {
        ref_mma16(0, out, a, b[0], out);
        ref_mma16(0, out, a, b[1], out);
        ref_mma16(0, out, a, b[2], out);
        ref_mma16(0, out, a, b[3], out);
    }
    ref_mma16(0, out + 4, a, b[0], out + 4);
}
// t_chain_exit: 4 products, then 4 more unless aux[1] is not zero (the whole warp leaves).
// in: a0 a1 (h) b0 b1 b2 b3 (h) c0..c3 (F). out: d0..d3 (F).
static void m_chain_exit(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[2]; memcpy(a, in, 2 * sizeof(reg));
    reg b[4]; memcpy(b, in + 2, 4 * sizeof(reg));
    memcpy(out, in + 6, 4 * sizeof(reg));
    for (int group = 0; group < (aux[1] ? 1 : 2); group++) {
        ref_mma16(0, out, a, b[0], out);
        ref_mma16(0, out, a, b[1], out);
        ref_mma16(0, out, a, b[2], out);
        ref_mma16(0, out, a, b[3], out);
    }
}
// t_chain_zero: six products on an accumulator that starts from zero.
// in: a0 a1 (h) b0 b1 b2 b3 (h). out: d0..d3 (F).
static void m_chain_zero(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    reg a[2]; memcpy(a, in, 2 * sizeof(reg));
    reg b[4]; memcpy(b, in + 2, 4 * sizeof(reg));
    memset(out, 0, 4 * sizeof(reg));
    static const int order[6] = {0, 1, 2, 3, 1, 2};
    for (int i = 0; i < 6; i++) ref_mma16(0, out, a, b[order[i]], out);
}
// t_chain_ld: operands from shared memory by two ldmatrix, two accumulators from zero, four
// products each in turn.
// in: rows of A, rows of B (o). out: d0..d3 e0..e3 (F).
static void m_chain_ld(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[4];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(4, 0, b, in[1], aux, 0);
    memset(out, 0, 8 * sizeof(reg));
    ref_mma16(0, out, a, b[0], out);         ref_mma16(0, out, a + 2, b[2], out);
    ref_mma16(0, out + 4, a, b[1], out + 4); ref_mma16(0, out + 4, a + 2, b[3], out + 4);
    ref_mma16(0, out, a, b[2], out);         ref_mma16(0, out, a + 2, b[0], out);
    ref_mma16(0, out + 4, a, b[3], out + 4); ref_mma16(0, out + 4, a + 2, b[1], out + 4);
}
// t_chain_loop: 4 turns of a chain of 4 products.
// in: a0 a1 (h) b0 b1 b2 b3 (h) c0..c3 (F). out: d0..d3 (F).
static void m_chain_loop(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    reg a[2]; memcpy(a, in, 2 * sizeof(reg));
    reg b[4]; memcpy(b, in + 2, 4 * sizeof(reg));
    memcpy(out, in + 6, 4 * sizeof(reg));
    for (int turn = 0; turn < 4; turn++) {
        ref_mma16(0, out, a, b[0], out);
        ref_mma16(0, out, a, b[1], out);
        ref_mma16(0, out, a, b[2], out);
        ref_mma16(0, out, a, b[3], out);
    }
}
static void m_loop(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[4], bt, swap;
    memcpy(a, in, sizeof a); memcpy(b, in + 4, sizeof b);
    reg* f = out; reg* h = out + 4; reg* n = out + 6;
    memcpy(f, in + 8, 4 * sizeof(reg)); memcpy(h, in + 12, 2 * sizeof(reg)); memcpy(n, in + 14, 2 * sizeof(reg));
    for (uint32_t step = 0; step < aux[0]; step++) {
        reg* aa = a + 2 * (step & 1);
        uint32_t* bb = b[step & 3];
        ref_mma16(0, f, aa, bb, f);
        ref_movmatrix(bt, bb);
        ref_mma16(1, h, aa, bt, h);
        ref_mma8(1, 1, n, aa[0], bb, n);
        memcpy(swap, a[0], sizeof swap); memcpy(a[0], a[2], sizeof swap); memcpy(a[2], swap, sizeof swap);
        memcpy(bb, bt, sizeof bt);
        ref_mma8(1, 1, n, a[0], bb, n);
    }
}

// Products that follow each other on one register of B (k_kept.cu): two, with accumulators in
// floats (0) or in halves (1); (2) two, then the register of B with its signs turned by those of
// the first result, then two more with the two A exchanged; (3) two into the same accumulators;
// (4) two with nothing to add to, and B given back.
static void m_kept(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    int f16 = v == 1, n = f16 ? 2 : 4;
    reg b;
    memcpy(b, in[4], sizeof b);
    if (v == 4) {
        memset(out, 0, 8 * sizeof(reg));
        ref_mma16(0, out, in, b, out);
        ref_mma16(0, out + 4, in + 2, b, out + 4);
        memcpy(out[8], b, sizeof b);
        return;
    }
    if (v == 3) {
        memcpy(out, in + 5, 4 * sizeof(reg));
        ref_mma16(0, out, in, b, out);
        ref_mma16(0, out, in + 2, b, out);
        return;
    }
    memcpy(out, in + 5, 2 * n * sizeof(reg));
    ref_mma16(f16, out, in, b, out);
    ref_mma16(f16, out + n, in + 2, b, out + n);
    if (v == 2) {
        for (int lane = 0; lane < 32; lane++) b[lane] ^= out[0][lane] & 0x80008000;
        ref_mma16(0, out, in + 2, b, out);
        ref_mma16(0, out + 4, in, b, out + 4);
    }
}

static void m_long(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    reg b;
    memcpy(b, in[2], sizeof b); memcpy(out, in + 3, 4 * sizeof(reg));
    for (int step = 0; step < 12; step++) {
        ref_mma16(0, out, in, b, out);
        reg t;
        ref_movmatrix(t, b);
        memcpy(b, t, sizeof b);
    }
}

// ---- k_groups.cu: GROUP patterns (2 A x 4 B = 8 products) -------------------------------------------
// Helper: 8 products of a GROUP (2 A x 4 B, f32 accumulators).
// d: 32 registers, a: 4 registers (A0=a[0:2], A1=a[2:4]), b: 4 registers.
static void group_8(reg* d, reg* a, reg* b) {
    ref_mma16(0, d + 0, a, b[0], d + 0);
    ref_mma16(0, d + 4, a, b[1], d + 4);
    ref_mma16(0, d + 8, a, b[2], d + 8);
    ref_mma16(0, d + 12, a, b[3], d + 12);
    ref_mma16(0, d + 16, a + 2, b[0], d + 16);
    ref_mma16(0, d + 20, a + 2, b[1], d + 20);
    ref_mma16(0, d + 24, a + 2, b[2], d + 24);
    ref_mma16(0, d + 28, a + 2, b[3], d + 28);
}
static void group_8_f16(reg* d, reg* a, reg* b) {
    ref_mma16(1, d + 0, a, b[0], d + 0);
    ref_mma16(1, d + 2, a, b[1], d + 2);
    ref_mma16(1, d + 4, a, b[2], d + 4);
    ref_mma16(1, d + 6, a, b[3], d + 6);
    ref_mma16(1, d + 8, a + 2, b[0], d + 8);
    ref_mma16(1, d + 10, a + 2, b[1], d + 10);
    ref_mma16(1, d + 12, a + 2, b[2], d + 12);
    ref_mma16(1, d + 14, a + 2, b[3], d + 14);
}
static void scale32(reg* d) {
    for (int i = 0; i < 32; i++) for (int lane = 0; lane < 32; lane++) {
        float f; memcpy(&f, &d[i][lane], 4); f *= 0.5f; memcpy(&d[i][lane], &f, 4);
    }
}

// g_once: one group, accumulators from zero.  in: o_a o_b; out: 32 F.
static void m_g_once(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[4];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(4, 0, b, in[1], aux, 0);
    memset(out, 0, 32 * sizeof(reg));
    group_8(out, a, b);
}

// g_loop: group in a loop over k.  variant = trip count.  in: n o_a o_b; out: 32 F.
static void m_g_loop(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    scale32(out);
}

// g_trans: as g_loop, B loaded by ldmatrix.trans.  in: n o_a o_b; out: 32 F.
static void m_g_trans(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 1, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    scale32(out);
}

// g_init: as g_loop, accumulators from input.  in: n o_a o_b d0..d31; out: 32 F.
static void m_g_init(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memcpy(out, in + 3, 32 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    scale32(out);
}

// g_peek: as g_loop with accumulator peek on even steps.  in: n o_a o_b; out: 34 F.
static void m_g_peek(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 34 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        // First 4 products (A0 x B0..B3)
        ref_mma16(0, out + 0, a, b[0], out + 0);
        ref_mma16(0, out + 4, a, b[1], out + 4);
        ref_mma16(0, out + 8, a, b[2], out + 8);
        ref_mma16(0, out + 12, a, b[3], out + 12);
        // Peek on even turns: read d[0] and d[16], add to output, flip d[0]
        if (step % 2 == 0) {
            for (int lane = 0; lane < 32; lane++) {
                float f0, f16, s;
                memcpy(&f0, &out[0][lane], 4);
                memcpy(&f16, &out[16][lane], 4);
                memcpy(&s, &out[32][lane], 4);
                s += f0; memcpy(&out[32][lane], &s, 4);
                memcpy(&s, &out[33][lane], 4);
                s += f16; memcpy(&out[33][lane], &s, 4);
                out[0][lane] ^= 0x80000000u;
            }
        }
        // Last 4 products (A1 x B0..B3)
        ref_mma16(0, out + 16, a + 2, b[0], out + 16);
        ref_mma16(0, out + 20, a + 2, b[1], out + 20);
        ref_mma16(0, out + 24, a + 2, b[2], out + 24);
        ref_mma16(0, out + 28, a + 2, b[3], out + 28);
    }
    scale32(out);
}

// g_partial: 6 of the 8 group products + 1 isolated.  in: o_a o_b o_bx; out: 28 F.
static void m_g_partial(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[4], bx[1];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(4, 0, b, in[1], aux, 0);
    ref_ldmatrix(1, 0, bx, in[2], aux, 0);
    memset(out, 0, 28 * sizeof(reg));
    ref_mma16(0, out + 0, a, b[0], out + 0);
    ref_mma16(0, out + 4, a, b[1], out + 4);
    ref_mma16(0, out + 8, a, b[2], out + 8);
    ref_mma16(0, out + 12, a, b[3], out + 12);
    ref_mma16(0, out + 16, a + 2, b[0], out + 16);
    ref_mma16(0, out + 20, a + 2, b[1], out + 20);
    ref_mma16(0, out + 24, a, bx[0], out + 24);
}

// g_shared: a load also read by an ordinary store.  in: o_a o_b; out: 32 F + 4 w.
static void m_g_shared(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[4];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(4, 0, b, in[1], aux, 0);
    memset(out, 0, 36 * sizeof(reg));
    group_8(out, a, b);
    for (int i = 0; i < 4; i++) memcpy(out[32 + i], a[i], sizeof(reg));
}

// g_two: two loops, divergent branch between.  in: n o_a o_b; out: 32 F.
static void m_g_two(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    // First loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    // Divergent branch: lane >= 16 multiplies by 2, others by 0.5
    for (int lane = 0; lane < 32; lane++) {
        float scale = lane >= 16 ? 2.0f : 0.5f;
        for (int i = 0; i < 32; i++) {
            float f; memcpy(&f, &out[i][lane], 4); f *= scale; memcpy(&out[i][lane], &f, 4);
        }
    }
    // Second loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
}

// g_blocks: loop with conditional extra loads on even turns.  in: n o_a o_b o_a2 o_b2; out: 32 F.
static void m_g_blocks(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4], a2[4], b2[4];
    memset(out, 0, 32 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
        if (step % 2 == 0) {
            ref_ldmatrix(4, 0, a2, in[3], aux, 0);
            ref_ldmatrix(4, 0, b2, in[4], aux, 0);
            group_8(out, a2, b2);
        }
    }
}

// g_after: divergent read of accumulators after loop.  in: n o_a o_b; out: 32 F.
static void m_g_after(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    // Divergent: lane < 16 multiplies by 2
    for (int lane = 0; lane < 32; lane++) {
        if (lane < 16) {
            for (int i = 0; i < 32; i++) {
                float f; memcpy(&f, &out[i][lane], 4); f *= 2.0f; memcpy(&out[i][lane], &f, 4);
            }
        }
    }
}

// g_sixteen: f16 accumulators, one group.  in: o_a o_b; out: 16 c.
static void m_g_sixteen(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[4];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(4, 0, b, in[1], aux, 0);
    memset(out, 0, 16 * sizeof(reg));
    group_8_f16(out, a, b);
}

// The kernels whose operands come from shared memory, which they fill with aux as it is.
static void m_ld_kq(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[2];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(2, 0, b, in[1], aux, 0);
    memcpy(out, in + 2, 4 * sizeof(reg));
    ref_mma16(0, out, a, b[0], out);
    ref_mma16(0, out, a + 2, b[1], out);
}
static void m_ld_vkq(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg b[2];
    ref_ldmatrix(2, 1, b, in[4], aux, 0);
    memcpy(out, in + 5, 2 * sizeof(reg));
    ref_mma16(1, out, in, b[0], out);
    ref_mma16(1, out, in + 2, b[1], out);
}
static void m_ld_s8(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[2];
    ref_ldmatrix(2, 0, a, in[0], aux, 0);
    memcpy(out, in + 3, 4 * sizeof(reg));
    ref_mma8(1, 1, out, a[0], in[1], out);
    ref_mma8(1, 1, out + 2, a[1], in[2], out + 2);
}
static void m_ld_mixed(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[2], b[1];
    ref_ldmatrix(2, 0, a, in[0], aux, 0);
    ref_ldmatrix(1, 0, b, in[1], aux, 0);
    memcpy(out, in + 3, 4 * sizeof(reg));
    ref_mma16(0, out, a, b[0], out);
    ref_mma16(0, out, a, in[2], out);
    memcpy(out[4], a[0], sizeof(reg)); memcpy(out[5], b[0], sizeof(reg));
}
static void m_ld_loop(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[2], b[2];
    ref_ldmatrix(2, 0, a, in[0], aux, 0);
    ref_ldmatrix(2, 0, b, in[1], aux, 0);
    memcpy(out, in + 3, 4 * sizeof(reg));
    for (uint32_t step = 0; step < in[2][0]; step++) {
        ref_mma16(0, out, a, b[0], out);
        ref_mma16(0, out, a, b[1], out);
    }
}

// ---- k_pairs.cu: PAIRS patterns (unconditional loads in loops) ------------------------------------------
// p_after: two loops, divergent branch between.  in: n o_a o_b; out: 32 F.
static void m_p_after(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    // First loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    // Divergent branch: lane < 16 multiplies by 2, others by 0.5
    for (int lane = 0; lane < 32; lane++) {
        float scale = lane < 16 ? 2.0f : 0.5f;
        for (int i = 0; i < 32; i++) {
            float f; memcpy(&f, &out[i][lane], 4); f *= scale; memcpy(&out[i][lane], &f, 4);
        }
    }
    // Second loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
}

// p_shared: load also read by ordinary XOR.  in: n o_a o_b; out: 36 (32 F + 4 w).
static void m_p_shared(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4], asum[4];
    memset(out, 0, 36 * sizeof(reg));
    memset(asum, 0, sizeof(asum));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        for (int i = 0; i < 4; i++)
            for (int lane = 0; lane < 32; lane++) asum[i][lane] ^= a[i][lane];
        group_8(out, a, b);
    }
    scale32(out);
    for (int i = 0; i < 4; i++) memcpy(out[32 + i], asum[i], sizeof(reg));
}

// p_multi: two loops with different loads.  in: n o_a1 o_b1 o_a2 o_b2; out: 32 F.
static void m_p_multi(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    // First loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    // Second loop with different addresses
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[3], aux, 0);
        ref_ldmatrix(4, 0, b, in[4], aux, 0);
        group_8(out, a, b);
    }
    scale32(out);
}

// p_call: call between two loops.  in: n o_a o_b; out: 32 F.
static void m_p_call(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    // First loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
    // Scale by 0.5 (what the noinline call does)
    scale32(out);
    // Second loop
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
    }
}

// p_uniform: loop with extra products on even turns.  in: n o_a o_b; out: 32 F.
static void m_p_uniform(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    int count = v;
    reg a[4], b[4];
    memset(out, 0, 32 * sizeof(reg));
    for (int step = 0; step < count; step++) {
        ref_ldmatrix(4, 0, a, in[1], aux, 0);
        ref_ldmatrix(4, 0, b, in[2], aux, 0);
        group_8(out, a, b);
        if (step % 2 == 0) {
            ref_ldmatrix(4, 0, a, in[1], aux, 0);
            ref_ldmatrix(4, 0, b, in[2], aux, 0);
            group_8(out, a, b);
        }
    }
    scale32(out);
}

// in/out: one letter per register: see value(); results are compared as f32 (F), f16 pairs (h, c) or bits.
// aux: 'h' if the shared memory must hold f16 values, any bits otherwise.
// ---- k_ampere.cu ----------------------------------------------------------------------------------------
static void m_k16(int f16, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    ref_mma16k16(f16, out, in, in + 4, in + 6);
}
static void m_ld_k16(int trans, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)block;
    reg a[4], b[2];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(2, trans, b, in[1], aux, 0);
    memcpy(out, in + 2, 4 * sizeof(reg));
    ref_mma16k16(0, out, a, b, out);
}
// Ampere mma.m16n8k16 on int8: A is 16x16 in two registers, B is 16x8 in one, C/D 16x8 in four s32.
// Written from the PTX manual's fragment layouts, not from the decomposition.
static void ref_mma16k16_int(int signed_a, int signed_b, int satfinite, reg* d, reg* a, uint32_t* b, reg* c) {
    int64_t A[16][16], B[16][8];
    int64_t D[16][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 8; i++) {
            uint8_t ab = a[i / 4][lane] >> (8 * (i & 3));
            int row = (i < 4) ? g : g + 8;
            A[row][4 * t + (i & 3)] = signed_a ? (int8_t)ab : ab;
        }
        for (int i = 0; i < 4; i++) {
            uint8_t bb = b[lane] >> (8 * i);
            B[4 * t + i][g] = signed_b ? (int8_t)bb : bb;
        }
        for (int i = 0; i < 4; i++) {
            int row = (i < 2) ? g : g + 8;
            int col = 2 * t + (i & 1);
            D[row][col] = (int32_t)c[i][lane];
        }
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < 16; k++) D[i][j] += A[i][k] * B[k][j];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) {
            int row = (i < 2) ? g : g + 8;
            int col = 2 * t + (i & 1);
            int64_t v = D[row][col];
            if (satfinite) v = v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : v);
            d[i][lane] = (uint32_t)(int32_t)v;
        }
    }
}
// Ampere mma.m16n8k32 on int8: A is 16x32 in four registers, B is 32x8 in two, C/D 16x8 in four s32.
static void ref_mma16k32_int(int signed_a, int signed_b, int satfinite, reg* d, reg* a, reg* b, reg* c) {
    int64_t A[16][32], B[32][8];
    int64_t D[16][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 16; i++) {
            uint8_t ab = a[i / 4][lane] >> (8 * (i & 3));
            int row = ((i < 4) || (i >= 8 && i < 12)) ? g : g + 8;
            int col = (i < 8) ? (4 * t + (i & 3)) : (4 * t + (i & 3) + 16);
            A[row][col] = signed_a ? (int8_t)ab : ab;
        }
        for (int i = 0; i < 8; i++) {
            uint8_t bb = b[i / 4][lane] >> (8 * (i & 3));
            int row = (i < 4) ? (4 * t + (i & 3)) : (4 * t + (i & 3) + 16);
            B[row][g] = signed_b ? (int8_t)bb : bb;
        }
        for (int i = 0; i < 4; i++) {
            int row = (i < 2) ? g : g + 8;
            int col = 2 * t + (i & 1);
            D[row][col] = (int32_t)c[i][lane];
        }
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < 32; k++) D[i][j] += A[i][k] * B[k][j];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) {
            int row = (i < 2) ? g : g + 8;
            int col = 2 * t + (i & 1);
            int64_t v = D[row][col];
            if (satfinite) v = v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : v);
            d[i][lane] = (uint32_t)(int32_t)v;
        }
    }
}
// Turing m8n8k16 with satfinite.
static void ref_mma8_sat(int signed_a, int signed_b, reg* d, uint32_t* a, uint32_t* b, reg* c) {
    int64_t A[8][16], B[16][8];
    int64_t D[8][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) {
            uint8_t ab = a[lane] >> (8 * i), bb = b[lane] >> (8 * i);
            A[g][4 * t + i] = signed_a ? (int8_t)ab : ab;
            B[4 * t + i][g] = signed_b ? (int8_t)bb : bb;
        }
        D[g][2 * t] = (int32_t)c[0][lane]; D[g][2 * t + 1] = (int32_t)c[1][lane];
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < 16; k++) D[i][j] += A[i][k] * B[k][j];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        int64_t v0 = D[g][2 * t], v1 = D[g][2 * t + 1];
        v0 = v0 > INT32_MAX ? INT32_MAX : (v0 < INT32_MIN ? INT32_MIN : v0);
        v1 = v1 > INT32_MAX ? INT32_MAX : (v1 < INT32_MIN ? INT32_MIN : v1);
        d[0][lane] = (uint32_t)(int32_t)v0; d[1][lane] = (uint32_t)(int32_t)v1;
    }
}

static void m_i8_k16(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    ref_mma16k16_int(v & 2, v & 1, 0, out, in, in[2], in + 3);
}
static void m_i8_k32(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_mma16k32_int(1, 1, 0, out, in, in + 4, in + 6);
}
static void m_i8_k16_sat(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_mma16k16_int(1, 1, 1, out, in, in[2], in + 3);
}
static void m_i8_k32_sat(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_mma16k32_int(1, 1, 1, out, in, in + 4, in + 6);
}
static void m_i8_ld_k32(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    memcpy(out, in + 3, 4 * sizeof(reg));
    ref_mma16k32_int(1, 1, 0, out, a, in + 1, out);
}
static void m_i8_turing_sat(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_mma8_sat(1, 1, out, in[0], in[1], in + 2);
}
static void m_i8_ld_k16(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[2];
    ref_ldmatrix(2, 0, a, in[0], aux, 0);
    memcpy(out, in + 2, 4 * sizeof(reg));
    ref_mma16k16_int(1, 1, 0, out, a, in[1], out);
}

// cp.async: the first `amount` bytes of the source, zeros after them.
static void m_cp(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        const uint8_t* source = (const uint8_t*)aux + in[0][lane];
        uint32_t amount = in[1][lane];
        uint8_t first[16] = {0};
        memcpy(first, source, amount);
        for (int word = 0; word < 4; word++) memcpy(&out[word][lane], first + 4 * word, 4);
        memcpy(&out[4][lane], source, 4);
        memcpy(&out[5][lane], source + 4, 4);
        out[6][lane] = 0;
        memcpy(&out[7][lane], source, 4);
    }
}
static void m_cp_guarded(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    for (int lane = 0; lane < 32; lane++) for (int word = 0; word < 4; word++)
        out[word][lane] = in[1][lane] & 1 ? aux[in[0][lane] / 4 + word] : 0xdddddddd;
}
static void m_pack(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) out[0][lane] = d2h(f2d(in[0][lane])) << 16 | d2h(f2d(in[1][lane]));
}
// max.NaN, min.NaN: a NaN if an input is one (the rewriting gives 0x7fffffff; the manual leaves the
// single-precision NaN unspecified), else the larger and the smaller, with -0 below +0.
static void m_nan(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint32_t x = in[0][lane], y = in[1][lane];
        double a = f2d(x), b = f2d(y);
        if (isnan(a) || isnan(b)) { out[0][lane] = out[1][lane] = 0x7fffffff; continue; }
        int a_is_larger = a > b || (a == b && !signbit(a));
        out[0][lane] = a_is_larger ? x : y;
        out[1][lane] = a_is_larger ? y : x;
    }
}

// The bf16 nearest to an f32 (ties to even) or toward zero: its upper half, the lower half deciding.
// A NaN gives 0x7fff, as the rewriting does.
static uint32_t bf16(uint32_t bits, int nearest) {
    if (isnan(f2d(bits))) return 0x7fff;
    uint32_t upper = bits >> 16, rest = bits & 0xffff;
    if (nearest && (rest > 0x8000 || (rest == 0x8000 && (upper & 1)))) upper++;
    return upper;
}
static void m_bf16(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint32_t x = in[0][lane], y = in[1][lane];
        out[0][lane] = bf16(x, 1);
        out[1][lane] = bf16(x, 0);
        out[2][lane] = bf16(x, 1) << 16 | bf16(y, 1);
        out[3][lane] = y & 0xffff0000;
    }
}
static void m_relu(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        int32_t a = (int32_t)in[0][lane], b = (int32_t)in[1][lane], low = a < b ? a : b, high = a < b ? b : a;
        out[0][lane] = low < 0 ? 0 : low;
        out[1][lane] = high < 0 ? 0 : high;
    }
}
// ---- products of bf16, tf32 and f64 -----------------------------------------------------------------
// The manual gives no order for the additions of a product. The layer adds the terms one after the
// other in the order of k, each with a fused multiply-add: the model does the same, so that the results
// can be compared bit for bit. (An RTX 5090 does not: it keeps more bits than an f32 while it adds and
// cuts the result toward zero. On values whose sums are not exact its results differ from these in
// their last bits, about one in six for bf16.) A tf32 operand is taken without the 13
// low bits of its mantissa, as the 5090 takes it. `elements` per register: 2 bf16 (each the upper half
// of an f32) or 1 tf32; `blocks` of k: one register of B for each, and two of A (rows g and g + 8).
static void ref_sums(int elements, int blocks, reg* d, reg* a, reg* b, reg* c) {
    enum { MAXK = 16 };
    float A[16][MAXK], B[MAXK][8], D[16][8];
    int per = 4 * elements, K = per * blocks;
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int q = 0; q < blocks; q++) for (int e = 0; e < elements; e++) {
            int k = q * per + t * elements + e;
            for (int h = 0; h < 2; h++) {
                uint32_t r = a[2 * q + h][lane], bits = elements == 2 ? (r >> (16 * e) & 0xffff) << 16 : r & 0xffffe000;
                memcpy(&A[g + 8 * h][k], &bits, 4);
            }
            uint32_t r = b[q][lane], bits = elements == 2 ? (r >> (16 * e) & 0xffff) << 16 : r & 0xffffe000;
            memcpy(&B[k][g], &bits, 4);
        }
        for (int i = 0; i < 4; i++) memcpy(&D[g + 8 * (i >> 1)][2 * t + (i & 1)], &c[i][lane], 4);
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < K; k++) D[i][j] = fmaf(A[i][k], B[k][j], D[i][j]);
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) memcpy(&d[i][lane], &D[g + 8 * (i >> 1)][2 * t + (i & 1)], 4);
    }
}
// variant: bit 0 = two blocks of k, bit 1 = bf16.
static void m_sums(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    int blocks = 1 + (v & 1);
    ref_sums(v & 2 ? 2 : 1, blocks, out, in, in + 2 * blocks, in + 3 * blocks);
}
static void m_bf16_twice(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_sums(2, 2, out, in, in + 8, in + 10);
    ref_sums(2, 2, out, in + 4, in + 8, out);
}
// mma.m8n8k4 on f64: A (one double: row g, column t), B (row t, column g), C and D (two doubles: row
// g, columns 2t and 2t + 1). The doubles are two words each.
static void m_f64_k4(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    double A[8][4], B[4][8], D[8][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        uint64_t word[4];
        for (int i = 0; i < 4; i++) word[i] = in[2 * i][lane] | (uint64_t)in[2 * i + 1][lane] << 32;
        memcpy(&A[g][t], &word[0], 8); memcpy(&B[t][g], &word[1], 8);
        memcpy(&D[g][2 * t], &word[2], 8); memcpy(&D[g][2 * t + 1], &word[3], 8);
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < 4; k++) D[i][j] = fma(A[i][k], B[k][j], D[i][j]);
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 2; i++) {
            uint64_t word;
            memcpy(&word, &D[g][2 * t + i], 8);
            out[2 * i][lane] = (uint32_t)word, out[2 * i + 1][lane] = word >> 32;
        }
    }
}
// cvt.rn.tf32.f32: the nearest f32 with 10 bits of mantissa, ties to even.
static void m_cvt_tf32(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        float f;
        memcpy(&f, &in[0][lane], 4);
        if (isnan(f) || isinf(f) || f == 0) { out[0][lane] = isnan(f) ? 0x7fc00000 : in[0][lane]; continue; }
        double a = fabs(f), r;
        int e;
        if (a < 0x1p-126) r = nearbyint(a * 0x1p136) * 0x1p-136;  // subnormal: the step does not shrink
        else { double m = frexp(a, &e); r = ldexp(nearbyint(ldexp(m, 11)), e - 11); }
        out[0][lane] = d2f(copysign(r, f));
    }
}

// ---- k_later.cu -------------------------------------------------------------------------------------
// a * b + c, a * b to nearest and towards minus infinity, a + c, a - c towards zero, on two pairs of
// floats: each float for itself. The operands are volatile so that the host computes when it runs,
// in the rounding mode set.
static void m_f32x2(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) for (int half = 0; half < 2; half++) {
        volatile float a, b, c, r[5];
        float value[3];
        for (int i = 0; i < 3; i++) memcpy(&value[i], &in[2 * i + half][lane], 4);
        a = value[0], b = value[1], c = value[2];
        r[0] = fmaf(a, b, c);
        r[1] = a * b;
        fesetround(FE_DOWNWARD);
        r[2] = a * b;
        fesetround(FE_TONEAREST);
        r[3] = a + c;
        fesetround(FE_TOWARDZERO);
        r[4] = a - c;
        fesetround(FE_TONEAREST);
        for (int i = 0; i < 5; i++) { float f = r[i]; memcpy(&out[2 * i + half][lane], &f, 4); }
    }
}
static void m_hint(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    for (int lane = 0; lane < 32; lane++) out[0][lane] = aux[(in[0][lane] & ~3u) / 4];
}

// ---- k_fp8.cu ---------------------------------------------------------------------------------------
// cvt.rn.f16x2.{e4m3x2,e5m2x2}: each byte as a half, the upper byte in the upper half. variant: e5m2.
static void m_f8_decode(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++)
        out[0][lane] = d2h(fp8_value(v, in[0][lane] & 0xff)) | d2h(fp8_value(v, in[0][lane] >> 8 & 0xff)) << 16;
}
// cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32 d, a, b: a in the upper byte. variant: e5m2.
static void m_f8_encode(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++)
        out[0][lane] = (uint32_t)fp8_nearest(v, f2d(in[0][lane])) << 8 | fp8_nearest(v, f2d(in[1][lane]));
}
// The same from the two halves of a register, to e4m3 and to e5m2.
static void m_f8_encode_halves(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) for (int e5m2 = 0; e5m2 < 2; e5m2++)
        out[e5m2][lane] = (uint32_t)fp8_nearest(e5m2, hi(in[0][lane])) << 8 | fp8_nearest(e5m2, lo(in[0][lane]));
}
// mma.m16n8k32 on FP8 with f32 accumulators, by the layouts of the manual: A four registers of four
// bytes (rows g and g + 8 for columns 4t..4t+3, then for 16+4t..), B two (rows 4t.., then 16+4t..,
// column g), C and D four f32 (rows g and g + 8, columns 2t and 2t + 1). Sums in double precision.
static void ref_fp8_mma(int a_e5m2, int b_e5m2, reg* d, reg* a, reg* b, reg* c) {
    double A[16][32], B[32][8], D[16][8];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 16; i++)
            A[i < 4 || (i >= 8 && i < 12) ? g : g + 8][4 * t + (i & 3) + (i >= 8 ? 16 : 0)] =
                fp8_value(a_e5m2, a[i / 4][lane] >> (8 * (i & 3)) & 0xff);
        for (int i = 0; i < 8; i++)
            B[4 * t + (i & 3) + (i >= 4 ? 16 : 0)][g] = fp8_value(b_e5m2, b[i / 4][lane] >> (8 * (i & 3)) & 0xff);
        for (int i = 0; i < 4; i++) D[i < 2 ? g : g + 8][2 * t + (i & 1)] = f2d(c[i][lane]);
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++)
        for (int k = 0; k < 32; k++) D[i][j] += A[i][k] * B[k][j];
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; i++) d[i][lane] = d2f(D[i < 2 ? g : g + 8][2 * t + (i & 1)]);
    }
}
// variant: bit 0 = B is e5m2, bit 1 = A is e5m2.
static void m_fp8_mma(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    ref_fp8_mma(v >> 1, v & 1, out, in, in + 4, in + 6);
}
static void m_fp8_mma_ld(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    reg a[4], b[2];
    ref_ldmatrix(4, 0, a, in[0], aux, 0);
    ref_ldmatrix(2, 0, b, in[1], aux, 0);
    memcpy(out, in + 2, 4 * sizeof(reg));
    ref_fp8_mma(0, 0, out, a, b, out);
}
static void m_fp8_mma_twice(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    ref_fp8_mma(0, 1, out, in, in + 8, in + 10);
    ref_fp8_mma(0, 1, out, in + 4, in + 8, out);
}

// ---- k_fp4.cu: E2M1 and UE8M0 conversions ----------------------------------------------------------
// (the data functions e2m1_value etc. are above the value() function; the model functions are here.)

// cvt.rn.f16x2.e2m1x2: decode two E2M1 nibbles to two halves.
static void m_fp4_decode_e2m1(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint8_t byte = in[0][lane] & 0xff;
        uint8_t lo_nibble = byte & 0xf, hi_nibble = (byte >> 4) & 0xf;
        out[0][lane] = d2h(e2m1_value(lo_nibble)) | (uint32_t)d2h(e2m1_value(hi_nibble)) << 16;
    }
}

// cvt.rn.satfinite.e2m1x2.f32: encode two f32 to packed E2M1 pair.
// a goes to the upper nibble, b to the lower nibble.
static void m_fp4_encode_e2m1(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint8_t upper = e2m1_nearest(f2d(in[0][lane]));
        uint8_t lower = e2m1_nearest(f2d(in[1][lane]));
        out[0][lane] = (uint32_t)(upper << 4 | lower);
    }
}

// cvt.{rp,rz}.satfinite.ue8m0x2.f32: encode two f32 to packed UE8M0 pair.
// variant: 0 = rp, 1 = rz.
static void m_fp4_encode_ue8m0(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)aux; (void)block;
    int rounding = v; // 0 = rn, 1 = rz, 2 = rp
    for (int lane = 0; lane < 32; lane++) {
        uint8_t upper = ue8m0_round(f2d(in[0][lane]), rounding);
        uint8_t lower = ue8m0_round(f2d(in[1][lane]), rounding);
        out[0][lane] = (uint32_t)(upper << 8 | lower);
    }
}

// cvt.rn.bf16x2.ue8m0x2: decode two UE8M0 bytes to two bf16 halves.
// Byte e maps to bf16 of 2^(e-127). e=0 → bf16 0x0040 (subnormal for 2^(-127)).
// e=0xff → bf16 0x7FFF (NaN with all mantissa bits). Verified against RTX 5090.
static void m_fp4_decode_ue8m0(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint16_t pair = in[0][lane] & 0xffff;
        uint8_t lo_byte = pair & 0xff, hi_byte = (pair >> 8) & 0xff;
        // ue8m0 to bf16: e << 7 for e 1..254; e=0 → 0x0040; e=0xff → 0x7FFF.
        uint32_t lo_bf16 = lo_byte == 0xff ? 0x7FFF : lo_byte == 0 ? 0x0040 : (uint32_t)lo_byte << 7;
        uint32_t hi_bf16 = hi_byte == 0xff ? 0x7FFF : hi_byte == 0 ? 0x0040 : (uint32_t)hi_byte << 7;
        out[0][lane] = lo_bf16 | (hi_bf16 << 16);
    }
}

static void m_wide(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)block;
    for (int lane = 0; lane < 32; lane++)
        for (int word = 0; word < 8; word++) out[word][lane] = aux[(in[0][lane] & ~31u) / 4 + word];
}
static void m_redux(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    uint32_t sum = 0;
    int32_t largest = INT32_MIN;
    for (int lane = 0; lane < 32; lane++) {
        sum += in[0][lane];
        if ((int32_t)in[0][lane] > largest) largest = (int32_t)in[0][lane];
    }
    for (int lane = 0; lane < 32; lane++) out[0][lane] = sum, out[1][lane] = (uint32_t)largest;
}

// ---- k_bf16.cu: bf16 arithmetic ----------------------------------------------------------------
// The sum of two doubles as a double and what its rounding left out: high + low is the sum exactly.
static void two_sum(double a, double b, double* high, double* low) {
    double sum = a + b, part = sum - a;
    *high = sum, *low = isfinite(sum) ? (a - (sum - part)) + (b - part) : 0;
}
// The sign of v - (m + l), where m + l is such an exact sum (m the double nearest to it): no other
// double lies between the two, so v is on the side of the sum that it is on of m.
static int beside(double v, double m, double l) { return v < m ? -1 : v > m ? 1 : l > 0 ? -1 : l < 0 ? 1 : 0; }
// The bf16 nearest to the real number high + low, ties to the even mantissa, by the definition: the
// largest bf16 not above its magnitude is searched among all of them, and that one or the next is
// taken, whichever is nearer. Beyond the largest by half a step or more: infinity.
static uint32_t bf16_nearest(double high, double low) {
    if (isnan(high)) return 0x7fc0;
    uint32_t sign = signbit(high) ? 0x8000 : 0, below = 0;
    double m = fabs(high), l = sign ? -low : low;
    if (isinf(m)) return sign | 0x7f80;
    for (uint32_t step = 0x4000; step; step >>= 1)
        if (below + step <= 0x7f80 && beside(bf16_magnitude(below + step), m, l) <= 0) below += step;
    if (below == 0x7f80) return sign | below;
    int halfway = beside((bf16_magnitude(below) + bf16_magnitude(below + 1)) / 2, m, l);
    return sign | (halfway > 0 ? below : halfway < 0 ? below + 1 : below + (below & 1));
}
static uint32_t bf16_sum(uint32_t a, uint32_t b) {
    double high, low;
    two_sum(bf16_value(a), bf16_value(b), &high, &low);
    return bf16_nearest(high, low);
}
// a * b + c rounded once (the product of two bf16 is a double exactly).
static uint32_t bf16_fma(uint32_t a, uint32_t b, uint32_t c) {
    double high, low;
    two_sum(bf16_value(a) * bf16_value(b), bf16_value(c), &high, &low);
    return bf16_nearest(high, low);
}
// b_arith: neg, add, mul, fma on bf16 in the low halves. b_arith_x2: add and fma on both halves.
static void m_b_arith(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint32_t a = in[0][lane] & 0xffff, b = in[1][lane] & 0xffff, c = in[2][lane] & 0xffff;
        out[0][lane] = a ^ 0x8000;
        out[1][lane] = bf16_sum(a, b);
        out[2][lane] = bf16_nearest(bf16_value(a) * bf16_value(b), 0);
        out[3][lane] = bf16_fma(a, b, c);
    }
}
static void m_b_arith_x2(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        out[0][lane] = out[1][lane] = 0;
        for (int half = 0; half < 2; half++) {
            uint32_t a = in[0][lane] >> (16 * half) & 0xffff, b = in[1][lane] >> (16 * half) & 0xffff;
            uint32_t c = in[2][lane] >> (16 * half) & 0xffff;
            out[0][lane] |= bf16_sum(a, b) << (16 * half);
            out[1][lane] |= bf16_fma(a, b, c) << (16 * half);
        }
    }
}
// b_cvt_f64: cvt.rn.bf16.f64 of the f64 in two words.
static void m_b_cvt_f64(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        uint64_t bits = in[0][lane] | (uint64_t)in[1][lane] << 32;
        double value;
        memcpy(&value, &bits, 8);
        out[0][lane] = bf16_nearest(value, 0);
    }
}
// b_setp: setp.eq.bf16 (1 or 0). b_set_x2: set.eq.bf16x2.bf16x2 (1.0 or 0.0 in each half).
static void m_b_setp(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) out[0][lane] = bf16_value(in[0][lane]) == bf16_value(in[1][lane]);
}
static void m_b_set_x2(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)v; (void)aux; (void)block;
    for (int lane = 0; lane < 32; lane++) {
        out[0][lane] = 0;
        for (int half = 0; half < 2; half++)
            if (bf16_value(in[0][lane] >> (16 * half)) == bf16_value(in[1][lane] >> (16 * half))) out[0][lane] |= 0x3f80u << (16 * half);
    }
}
// b_atom_*: every thread of a block (variant 0), or every other one (variant 1), adds aux[1] to a
// word that holds aux[0] (see k_bf16.cu). The same in every thread: how many of the partial sums were
// given back to exactly one thread (all), what the word holds in the end, and how many of the threads
// that were not to add still have what they had. Even variants: one bf16; odd ones: two.
static void m_b_atom(int v, reg* out, reg* in, const uint32_t* aux, int block) {
    (void)in; (void)block;
    int adding = v & 1 ? BLOCK / 2 : BLOCK;
    uint32_t held = 0;
    for (int half = 0; half < (v & 2 ? 2 : 1); half++) {
        double sum = bf16_value(aux[0] >> (16 * half));
        for (int i = 0; i < adding; i++) sum += bf16_value(aux[1] >> (16 * half));
        held |= (d2f(sum) >> 16) << (16 * half);
    }
    for (int lane = 0; lane < 32; lane++) out[0][lane] = adding, out[1][lane] = held, out[2][lane] = BLOCK - adding;
}

static const struct test {
    const char* name; const char* in; const char* out; model_fn* model; int variant; char aux;
} tests[] = {
    {"v884_rowrow_f32f32", "hhhhFFFFFFFF", "FFFFFFFF", m_v884, 0}, {"v884_rowcol_f32f32", "hhhhFFFFFFFF", "FFFFFFFF", m_v884, 1},
    {"v884_colrow_f32f32", "hhhhFFFFFFFF", "FFFFFFFF", m_v884, 2}, {"v884_colcol_f32f32", "hhhhFFFFFFFF", "FFFFFFFF", m_v884, 3},
    {"v884_rowrow_f16f16", "hhhhcccc", "cccc", m_v884, 4}, {"v884_rowcol_f16f16", "hhhhcccc", "cccc", m_v884, 5},
    {"v884_colrow_f16f16", "hhhhcccc", "cccc", m_v884, 6}, {"v884_colcol_f16f16", "hhhhcccc", "cccc", m_v884, 7},
    {"v884_rowrow_f32f16", "hhhhcccc", "FFFFFFFF", m_v884, 8}, {"v884_rowcol_f32f16", "hhhhcccc", "FFFFFFFF", m_v884, 9},
    {"v884_colrow_f32f16", "hhhhcccc", "FFFFFFFF", m_v884, 10}, {"v884_colcol_f32f16", "hhhhcccc", "FFFFFFFF", m_v884, 11},
    {"t_mma_f32", "hhhFFFF", "FFFF", m_mma16, 0}, {"t_mma_f16", "hhhcc", "cc", m_mma16, 1},
    {"t_mma_s8s8", "wwww", "ww", m_mma8, 3}, {"t_mma_u8s8", "wwww", "ww", m_mma8, 1},
    {"t_mma_s8u8", "wwww", "ww", m_mma8, 2}, {"t_mma_u8u8", "wwww", "ww", m_mma8, 0},
    {"t_movmatrix", "w", "w", m_movmatrix, 0},
    {"t_ldmatrix_x1", "o", "w", m_ldmatrix, 1}, {"t_ldmatrix_x2", "o", "ww", m_ldmatrix, 2},
    {"t_ldmatrix_x4", "o", "wwww", m_ldmatrix, 4}, {"t_ldmatrix_x1_trans", "o", "w", m_ldmatrix, 9},
    {"t_ldmatrix_x2_trans", "o", "ww", m_ldmatrix, 10}, {"t_ldmatrix_x4_trans", "o", "wwww", m_ldmatrix, 12},
    {"t_ldmatrix_x4_shared32", "o", "wwww", m_ldmatrix, 4}, {"t_ldmatrix_x2_trans_shared64", "o", "ww", m_ldmatrix, 10},
    {"t_chain_f32", "hhhhhhhhFFFFFFFF", "FFFFFFFF", m_chain_f, 0}, {"t_chain_f16", "hhhhhhhhcccc", "cccc", m_chain_f, 1},
    {"t_chain_s8", "wwwwwwwwww", "wwww", m_chain_s8, 0},
    {"t_chain_end_read", "hhhhhhFFFFF", "FFFF", m_chain_end_read, 0},
    {"t_chain_carry", "hhhhhhFFFFF", "FFFF", m_chain_carry, 0},
    {"t_chain_loop", "hhhhhhFFFF", "FFFF", m_chain_loop, 0},
    {"t_chain_back", "hhhhhhFFFFFFFF", "FFFFFFFF", m_chain_back, 0},
    {"t_chain_exit", "hhhhhhFFFF", "FFFF", m_chain_exit, 0},
    {"t_chain_zero", "hhhhhh", "FFFF", m_chain_zero, 0},
    {"t_chain_ld", "oo", "FFFFFFFF", m_chain_ld, 0, 'h'},
    {"t_kept_f32", "hhhhhFFFFFFFF", "FFFFFFFF", m_kept, 0}, {"t_kept_f16", "hhhhhcccc", "cccc", m_kept, 1},
    {"t_kept_again", "hhhhhFFFFFFFF", "FFFFFFFF", m_kept, 2}, {"t_kept_chain", "hhhhhFFFF", "FFFF", m_kept, 3},
    {"t_kept_zero", "hhhhh", "FFFFFFFFh", m_kept, 4},
    {"t_loop", "hhhhhhhhFFFFccww", "FFFFccww", m_loop, 0}, {"t_long", "hhhFFFF", "FFFF", m_long, 0},
    {"t_ld_kq", "ooFFFF", "FFFF", m_ld_kq, 0, 'h'}, {"t_ld_kq", "uuFFFF", "FFFF", m_ld_kq, 0, 'h'},
    {"t_ld_vkq", "hhhhocc", "cc", m_ld_vkq, 0, 'h'}, {"t_ld_vkq", "hhhhucc", "cc", m_ld_vkq, 0, 'h'},
    {"t_ld_s8", "owwwwww", "wwww", m_ld_s8, 0, 'w'}, {"t_ld_s8", "uwwwwww", "wwww", m_ld_s8, 0, 'w'},
    {"t_ld_mixed", "oohFFFF", "FFFFww", m_ld_mixed, 0, 'h'}, {"t_ld_mixed", "uuhFFFF", "FFFFww", m_ld_mixed, 0, 'h'},
    {"t_ld_loop", "oonFFFF", "FFFF", m_ld_loop, 0, 'h'}, {"t_ld_loop", "uunFFFF", "FFFF", m_ld_loop, 0, 'h'},
    {"a_mma_f32", "hhhhhhFFFF", "FFFF", m_k16, 0}, {"a_mma_f16", "hhhhhhcc", "cc", m_k16, 1},
    {"a_ld_k16", "ooFFFF", "FFFF", m_ld_k16, 0, 'h'}, {"a_ld_k16", "uuFFFF", "FFFF", m_ld_k16, 0, 'h'},
    {"a_ld_k16_trans", "ooFFFF", "FFFF", m_ld_k16, 1, 'h'}, {"a_ld_k16_trans", "uuFFFF", "FFFF", m_ld_k16, 1, 'h'},
    {"a_i8_k16", "wwwwwww", "wwww", m_i8_k16, 3}, {"a_u8s8_k16", "wwwwwww", "wwww", m_i8_k16, 1},
    {"a_i8_k32", "wwwwwwwwww", "wwww", m_i8_k32, 0},
    {"a_i8_k16_sat", "wwwwwww", "wwww", m_i8_k16_sat, 0}, {"a_i8_k16_sat", "wwwSSSS", "wwww", m_i8_k16_sat, 0},
    {"a_i8_turing_sat", "wwww", "ww", m_i8_turing_sat, 0}, {"a_i8_turing_sat", "wwSS", "ww", m_i8_turing_sat, 0},
    {"a_i8_k32_sat", "wwwwwwwwww", "wwww", m_i8_k32_sat, 0}, {"a_i8_k32_sat", "wwwwwwSSSS", "wwww", m_i8_k32_sat, 0},
    {"a_i8_ld_k32", "owwwwww", "wwww", m_i8_ld_k32, 0, 'w'}, {"a_i8_ld_k32", "uwwwwww", "wwww", m_i8_ld_k32, 0, 'w'},
    {"a_i8_ld_k16", "owwwww", "wwww", m_i8_ld_k16, 0, 'w'}, {"a_i8_ld_k16", "uwwwww", "wwww", m_i8_ld_k16, 0, 'w'},
    {"s_bf16_k16", "BBBBBBFFFF", "XXXX", m_sums, 3}, {"s_bf16_k8", "BBBFFFF", "XXXX", m_sums, 2},
    {"s_tf32_k8", "TTTTTTFFFF", "XXXX", m_sums, 1}, {"s_tf32_k4", "TTTFFFF", "XXXX", m_sums, 0},
    {"s_f64_k4", "LHLHLHLH", "wwww", m_f64_k4, 0}, {"s_bf16_twice", "BBBBBBBBBBFFFF", "XXXX", m_bf16_twice, 0},
    {"s_cvt_tf32", "N", "X", m_cvt_tf32, 0}, {"s_cvt_tf32", "T", "X", m_cvt_tf32, 0},
    // ---- k_later.cu
    {"l_f32x2", "NNNNNN", "XXXXXXXXXX", m_f32x2, 0}, {"l_f32x2", "TTTTTT", "XXXXXXXXXX", m_f32x2, 0},
    {"l_hint", "o", "w", m_hint, 0},
    // ---- k_fp8.cu
    {"f8_decode_e4m3", "e", "n", m_f8_decode, 0}, {"f8_decode_e5m2", "e", "n", m_f8_decode, 1},
    {"f8_encode_e4m3", "xx", "4", m_f8_encode, 0}, {"f8_encode_e4m3", "NN", "4", m_f8_encode, 0},
    {"f8_encode_e5m2", "yy", "5", m_f8_encode, 1}, {"f8_encode_e5m2", "NN", "5", m_f8_encode, 1},
    {"f8_encode_halves", "w", "45", m_f8_encode_halves, 0},
    {"f8_mma_e4m3_e4m3", "EEEEEEFFFF", "PPPP", m_fp8_mma, 0}, {"f8_mma_e4m3_e4m3", "QQQQJJAAAA", "XXXX", m_fp8_mma, 0},
    {"f8_mma_e4m3_e4m3", "JJJJQQAAAA", "XXXX", m_fp8_mma, 0}, {"f8_mma_e4m3_e4m3", "GGGGJJIIII", "XXXX", m_fp8_mma, 0},
    {"f8_mma_e5m2_e4m3", "WWWWJJIIII", "XXXX", m_fp8_mma, 2}, {"f8_mma_e4m3_e5m2", "JJJJWWIIII", "XXXX", m_fp8_mma, 1},
    {"f8_mma_e5m2_e4m3", "MMMMEEFFFF", "PPPP", m_fp8_mma, 2}, {"f8_mma_e5m2_e4m3", "RRRRJJAAAA", "XXXX", m_fp8_mma, 2},
    {"f8_mma_e4m3_e5m2", "EEEEMMFFFF", "PPPP", m_fp8_mma, 1}, {"f8_mma_e4m3_e5m2", "JJJJRRAAAA", "XXXX", m_fp8_mma, 1},
    {"f8_mma_ld_e4m3", "ooFFFF", "PPPP", m_fp8_mma_ld, 0, 'E'}, {"f8_mma_twice", "EEEEEEEEMMFFFF", "PPPP", m_fp8_mma_twice, 0},
    // ---- k_fp4.cu: E2M1 and UE8M0 conversions
    {"fp4_decode_e2m1", "f", "n", m_fp4_decode_e2m1, 0}, {"fp4_decode_e2m1", "w", "n", m_fp4_decode_e2m1, 0},
    {"fp4_encode_e2m1", "11", "6", m_fp4_encode_e2m1, 0}, {"fp4_encode_e2m1", "NN", "6", m_fp4_encode_e2m1, 0},
    {"fp4_encode_ue8m0_rp", "22", "8", m_fp4_encode_ue8m0, 2}, {"fp4_encode_ue8m0_rp", "NN", "8", m_fp4_encode_ue8m0, 2},
    {"fp4_encode_ue8m0_rz", "33", "8", m_fp4_encode_ue8m0, 1}, {"fp4_encode_ue8m0_rz", "NN", "8", m_fp4_encode_ue8m0, 1},
    {"fp4_decode_ue8m0", "U", "Z", m_fp4_decode_ue8m0, 0}, {"fp4_decode_ue8m0", "w", "Z", m_fp4_decode_ue8m0, 0},
    {"a_cp", "oz", "wwwwwwww", m_cp, 0}, {"a_cp_guarded", "ow", "wwww", m_cp_guarded, 0}, {"a_pack", "gg", "w", m_pack, 0}, {"a_nan", "NN", "ww", m_nan, 0},
    {"a_bf16", "NN", "wwww", m_bf16, 0}, {"a_relu", "ww", "ww", m_relu, 0},
    {"a_wide", "o", "wwwwwwww", m_wide, 0}, {"a_redux", "w", "ww", m_redux, 0},
    // ---- k_bf16.cu: bf16 arithmetic
    {"b_arith", "dpq", "bbbb", m_b_arith, 0}, {"b_arith_x2", "rst", "ZZ", m_b_arith_x2, 0},
    {"b_arith_any", "mmm", "bbbb", m_b_arith, 0}, {"b_arith_x2_any", "BBB", "ZZ", m_b_arith_x2, 0},
    {"b_cvt_f64", "DH", "b", m_b_cvt_f64, 0}, {"b_setp", "mm", "w", m_b_setp, 0}, {"b_set_x2", "BB", "Z", m_b_set_x2, 0},
    {"b_atom", "w", "www", m_b_atom, 0, 'a'}, {"b_atom_guarded", "w", "www", m_b_atom, 1, 'a'},
    {"b_atom_shared", "w", "www", m_b_atom, 0, 'a'}, {"b_atom_shared_guarded", "w", "www", m_b_atom, 1, 'a'},
    {"b_atom_x2", "w", "www", m_b_atom, 2, 'v'}, {"b_atom_x2_guarded", "w", "www", m_b_atom, 3, 'v'},
    {"b_atom_x2_shared", "w", "www", m_b_atom, 2, 'v'}, {"b_atom_x2_shared_guarded", "w", "www", m_b_atom, 3, 'v'},
    // ---- k_groups.cu: GROUP patterns (2 A x 4 B = 8 products)
#define F32 "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    {"g_once", "oo", F32, m_g_once, 0, 'h'},
    {"g_loop", "noo", F32, m_g_loop, 0, 'h'}, {"g_loop", "noo", F32, m_g_loop, 1, 'h'},
    {"g_loop", "noo", F32, m_g_loop, 2, 'h'}, {"g_loop", "noo", F32, m_g_loop, 5, 'h'},
    {"g_trans", "noo", F32, m_g_trans, 0, 'h'}, {"g_trans", "noo", F32, m_g_trans, 1, 'h'},
    {"g_trans", "noo", F32, m_g_trans, 2, 'h'}, {"g_trans", "noo", F32, m_g_trans, 5, 'h'},
    {"g_init", "noo" F32, F32, m_g_init, 1, 'h'}, {"g_init", "noo" F32, F32, m_g_init, 2, 'h'},
    {"g_init", "noo" F32, F32, m_g_init, 5, 'h'},
    {"g_peek", "noo", F32 "FF", m_g_peek, 2, 'h'}, {"g_peek", "noo", F32 "FF", m_g_peek, 5, 'h'},
    {"g_partial", "ooo", "FFFFFFFFFFFFFFFFFFFFFFFFFFFF", m_g_partial, 0, 'h'},
    {"g_shared", "oo", F32 "wwww", m_g_shared, 0, 'h'},
    {"g_two", "noo", F32, m_g_two, 1, 'h'}, {"g_two", "noo", F32, m_g_two, 2, 'h'},
    {"g_two", "noo", F32, m_g_two, 5, 'h'},
    {"g_blocks", "noooo", F32, m_g_blocks, 1, 'h'}, {"g_blocks", "noooo", F32, m_g_blocks, 2, 'h'},
    {"g_blocks", "noooo", F32, m_g_blocks, 5, 'h'},
    {"g_after", "noo", F32, m_g_after, 1, 'h'}, {"g_after", "noo", F32, m_g_after, 2, 'h'},
    {"g_after", "noo", F32, m_g_after, 5, 'h'},
    {"g_sixteen", "oo", "cccccccccccccccc", m_g_sixteen, 0, 'h'},
    // ---- k_pairs.cu: PAIRS patterns (unconditional loads, pairs of products on whole rows)
    // p_basic: simple loop, unconditional loads.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_basic", "noo", F32, m_g_loop, 0, 'h'}, {"p_basic", "noo", F32, m_g_loop, 1, 'h'},
    {"p_basic", "noo", F32, m_g_loop, 2, 'h'}, {"p_basic", "noo", F32, m_g_loop, 5, 'h'},
    // p_trans: B loaded transposed.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_trans", "noo", F32, m_g_trans, 0, 'h'}, {"p_trans", "noo", F32, m_g_trans, 1, 'h'},
    {"p_trans", "noo", F32, m_g_trans, 2, 'h'}, {"p_trans", "noo", F32, m_g_trans, 5, 'h'},
    // p_pipe: pipelined double-buffer.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_pipe", "noo", F32, m_g_loop, 0, 'h'}, {"p_pipe", "noo", F32, m_g_loop, 1, 'h'},
    {"p_pipe", "noo", F32, m_g_loop, 2, 'h'}, {"p_pipe", "noo", F32, m_g_loop, 5, 'h'},
    // p_init: accumulators from input.  variant = trip count.  in: n o_a o_b d0..d31; out: 32 F.
    {"p_init", "noo" F32, F32, m_g_init, 1, 'h'}, {"p_init", "noo" F32, F32, m_g_init, 2, 'h'},
    {"p_init", "noo" F32, F32, m_g_init, 5, 'h'},
    // p_peek: accumulators read between turns.  variant = trip count.  in: n o_a o_b; out: 34 F.
    {"p_peek", "noo", F32 "FF", m_g_peek, 2, 'h'}, {"p_peek", "noo", F32 "FF", m_g_peek, 5, 'h'},
    // p_after: divergent read + two loops.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_after", "noo", F32, m_p_after, 1, 'h'}, {"p_after", "noo", F32, m_p_after, 2, 'h'},
    {"p_after", "noo", F32, m_p_after, 5, 'h'},
    // p_shared: load also read by ordinary instructions.  variant = trip count.  in: n o_a o_b; out: 36.
    {"p_shared", "noo", F32 "wwww", m_p_shared, 1, 'h'}, {"p_shared", "noo", F32 "wwww", m_p_shared, 2, 'h'},
    {"p_shared", "noo", F32 "wwww", m_p_shared, 5, 'h'},
    // p_multi: two loops with different loads.  variant = trip count.  in: n o_a1 o_b1 o_a2 o_b2; out: 32 F.
    {"p_multi", "noooo", F32, m_p_multi, 1, 'h'}, {"p_multi", "noooo", F32, m_p_multi, 2, 'h'},
    {"p_multi", "noooo", F32, m_p_multi, 5, 'h'},
    // p_call: call between two loops.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_call", "noo", F32, m_p_call, 1, 'h'}, {"p_call", "noo", F32, m_p_call, 2, 'h'},
    {"p_call", "noo", F32, m_p_call, 5, 'h'},
    // p_callee: products in a callee.  in: o_a o_b; out: 32 F.
    {"p_callee", "oo", F32, m_g_once, 0, 'h'},
    // p_uniform: uniform branch with extra products.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_uniform", "noo", F32, m_p_uniform, 1, 'h'}, {"p_uniform", "noo", F32, m_p_uniform, 2, 'h'},
    {"p_uniform", "noo", F32, m_p_uniform, 5, 'h'},
    // p_lead: syncthreads in the loop.  variant = trip count.  in: n o_a o_b; out: 32 F.
    {"p_lead", "noo", F32, m_g_loop, 0, 'h'}, {"p_lead", "noo", F32, m_g_loop, 1, 'h'},
    {"p_lead", "noo", F32, m_g_loop, 2, 'h'}, {"p_lead", "noo", F32, m_g_loop, 5, 'h'},
#undef F32
};

static int verbose, aligned_only;
static int wanted(const char* kernel) {
    const char* list = getenv("TCRUN_KERNELS");
    if (!list) return 1;
    size_t n = strlen(kernel);
    for (const char* p = list; (p = strstr(p, kernel)); p += n)
        if ((p == list || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 1;
    return 0;
}

// Is `got` wrong for a register of this kind whose model value is `want`?
static int wrong(char kind, int mode, uint32_t got, uint32_t want) {
    if (kind == 'X' || kind == 'P') {
        // An f32 equal to the model's, a NaN for a NaN; P, on random values: within the rounding of
        // an accumulation in an order the model does not know (sums below 2,048, terms above 1/64).
        double g = f2d(got), w = f2d(want);
        if (isnan(g) || isnan(w)) return !(isnan(g) && isnan(w));
        if (got == want || g == w) return 0;
        return !(kind == 'P' && mode == 1 && fabs(g - w) <= 2e-3 + 1e-5 * fabs(w));
    }
    if (kind == 'n') {
        // Two halves, each equal to the model's or a NaN for a NaN.
        for (int half = 0; half < 2; half++) {
            uint32_t g = got >> (16 * half) & 0xffff, w = want >> (16 * half) & 0xffff;
            if (g != w && !(isnan(h2d(g)) && isnan(h2d(w)))) return 1;
        }
        return 0;
    }
    if (kind == '4' || kind == '5') {
        // Two FP8 bytes (and nothing above them), each equal to the model's or a NaN for a NaN.
        if (got >> 16) return 1;
        for (int byte = 0; byte < 2; byte++) {
            uint8_t g = got >> (8 * byte), w = want >> (8 * byte);
            if (g != w && !(fp8_is_nan(kind == '5', g) && fp8_is_nan(kind == '5', w))) return 1;
        }
        return 0;
    }
    if (kind == 'b') {
        // A bf16 value in the low 16 bits of a 32-bit word, the upper bits zero; a NaN for a NaN.
        uint32_t g = got & 0xffff, w = want & 0xffff;
        if (got >> 16) return 1;  // upper bits should be zero
        int g_nan = (g & 0x7f80) == 0x7f80 && (g & 0x7f) != 0;  // bf16 NaN: exponent all 1s, mantissa != 0
        int w_nan = (w & 0x7f80) == 0x7f80 && (w & 0x7f) != 0;
        if (g_nan || w_nan) return !(g_nan && w_nan);
        return g != w;
    }
    if (kind == '6') {
        // A packed E2M1 byte (two nibbles) in the low 8 bits. E2M1 has no NaN: exact match.
        return (got & 0xff) != (want & 0xff);
    }
    if (kind == '8') {
        // A packed UE8M0 pair (two bytes) in the low 16 bits. 0xff is NaN.
        if (got >> 16) return 1;
        for (int byte = 0; byte < 2; byte++) {
            uint8_t g = (got >> (8 * byte)) & 0xff, w = (want >> (8 * byte)) & 0xff;
            if (g == w) continue;
            if (g == 0xff || w == 0xff) {
                if (g != 0xff || w != 0xff) return 1; // both must be NaN
            } else return 1;
        }
        return 0;
    }
    if (kind == 'Z') {
        // Two bf16 values in a 32-bit word; each must match, a NaN for a NaN.
        for (int half = 0; half < 2; half++) {
            uint32_t g = (got >> (16 * half)) & 0xffff, w = (want >> (16 * half)) & 0xffff;
            int g_nan = (g & 0x7f80) == 0x7f80 && (g & 0x7f) != 0;
            int w_nan = (w & 0x7f80) == 0x7f80 && (w & 0x7f) != 0;
            if (g_nan || w_nan) { if (!(g_nan && w_nan)) return 1; continue; }
            if (g != w) return 1;
        }
        return 0;
    }
    if (kind == 'F') {
        double g = f2d(got), w = f2d(want);
        return mode == 0 ? !(g == w) : !(fabs(g - w) <= 1e-4 * (1 + fabs(w)));
    }
    if (kind == 'h' || kind == 'c') {
        for (int half = 0; half < 2; half++) {
            double g = h2d(got >> (16 * half) & 0xffff), w = h2d(want >> (16 * half) & 0xffff);
            // A few units in the last place of the larger partial sums.
            if (mode == 0 ? !(g == w) : !(fabs(g - w) <= ldexp(fmax(fabs(w), 4), -7))) return 1;
        }
        return 0;
    }
    return got != want;
}

static int run(CUmodule module, const struct test* test, CUdeviceptr d_in, CUdeviceptr d_out, CUdeviceptr d_aux) {
    CUfunction function;
    if (cuModuleGetFunction(&function, module, test->name)) return -1;
    int nin = strlen(test->in), nout = strlen(test->out);
    static uint32_t in[THREADS * MAXREG], out[THREADS * MAXREG], aux[AUX_WORDS];
    long bad = 0, total = 0;
    for (int mode = 0; mode < 2; mode++) {
        for (int i = 0; i < AUX_WORDS; i++) aux[i] = test->aux && test->aux != 'w' ? value(test->aux, mode) : rnd();
        if (test->model == m_loop) aux[0] = LOOP_STEPS;
        if (test->model == m_chain_back) aux[0] = CHAIN_TURNS;
        if (test->model == m_chain_exit) aux[1] = (uint32_t)mode;    // (stays the first time, leaves the second)
        for (int thread = 0; thread < THREADS; thread++)
            for (int r = 0; r < nin; r++) in[thread * nin + r] = value(test->in[r], mode);
        // Group loop kernels: the trip count (input register 0, format 'n') is the variant.
        if (test->model == m_g_loop || test->model == m_g_trans || test->model == m_g_init ||
            test->model == m_g_peek || test->model == m_g_two || test->model == m_g_blocks ||
            test->model == m_g_after ||
            test->model == m_p_after || test->model == m_p_shared || test->model == m_p_multi ||
            test->model == m_p_call || test->model == m_p_uniform) {
            for (int thread = 0; thread < THREADS; thread++) in[thread * nin] = test->variant;
        }
        CK(cuMemcpyHtoD_v2(d_in, in, sizeof in));
        CK(cuMemcpyHtoD_v2(d_aux, aux, sizeof aux));
        CK(cuMemsetD8_v2(d_out, 0xee, sizeof out));
        void* args[] = {&d_in, &d_out, &d_aux};
        CK(cuLaunchKernel(function, GRID, 1, 1, BLOCK, 1, 1, 0, NULL, args, NULL));
        CUresult r = cuCtxSynchronize();
        if (r) { printf("FAIL %s: launch -> %d %s\n", test->name, r, ename(r)); exit(1); }
        CK(cuMemcpyDtoH_v2(out, d_out, sizeof out));
        for (int warp = 0; warp < WARPS; warp++) {
            reg rin[MAXREG], rout[MAXREG];
            for (int lane = 0; lane < 32; lane++)
                for (int r = 0; r < nin; r++) rin[r][lane] = in[(warp * 32 + lane) * nin + r];
            test->model(test->variant, rout, rin, aux, warp * 32 / BLOCK);
            for (int lane = 0; lane < 32; lane++) for (int r = 0; r < nout; r++) {
                uint32_t got = out[(warp * 32 + lane) * nout + r], want = rout[r][lane];
                total++;
                if (!wrong(test->out[r], mode, got, want)) continue;
                if (bad++ < 8 || verbose)
                    printf("    %s %s: warp %d lane %2d register %d: got %08x, model %08x\n", test->name,
                           mode ? "random" : "exact", warp, lane, r, got, want);
            }
        }
    }
    char label[64];
    snprintf(label, sizeof label, "%s%s", test->name, strchr(test->in, 'u') ? ", rows not aligned" : "");
    printf("  %-28s %s  (%ld/%ld wrong)\n", label, bad ? "WRONG" : "ok", bad, total);
    return bad != 0;
}

// -m COUNT: print what the bf16 models answer for COUNT sets of operands (those of the tests, both
// modes), for bf16_models.py to compare with exact arithmetic. No GPU.
static int print_models(int count) {
    for (int i = 0; i < count; i++) {
        uint32_t abc[3];
        bf16_operands(i & 1, abc);
        printf("fma %04x %04x %04x %04x\n", abc[0], abc[1], abc[2], bf16_fma(abc[0], abc[1], abc[2]));
        printf("add %04x %04x %04x\n", abc[0], abc[2], bf16_sum(abc[0], abc[2]));
        printf("mul %04x %04x %04x\n", abc[0], abc[1], bf16_nearest(bf16_value(abc[0]) * bf16_value(abc[1]), 0));
        uint64_t bits = value('D', i & 1);
        bits |= (uint64_t)value('H', i & 1) << 32;
        double v;
        memcpy(&v, &bits, 8);
        printf("cvt %016llx %04x\n", (unsigned long long)bits, bf16_nearest(v, 0));
    }
    return 0;
}

int main(int argc, char** argv) {
    int ordinal = 0, a = 1;
    for (; a < argc && argv[a][0] == '-'; a++) {
        if (!strcmp(argv[a], "-m")) return print_models(atoi(argv[a + 1]));
        else if (!strcmp(argv[a], "-d")) ordinal = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-s")) rng_state ^= strtoull(argv[++a], NULL, 0) * 0x2545F4914F6CDD1Dull;
        else if (!strcmp(argv[a], "-v")) verbose = 1;
        else if (!strcmp(argv[a], "-a")) aligned_only = 1;
    }
    if (a >= argc) { fprintf(stderr, "usage: tcrun [-d ORDINAL] [-s SEED] [-v] [-a] MODULE_FILE\n"); return 2; }
    const char* path = argv[a];
    FILE* fp = fopen(path, "rb");
    if (!fp) { perror(path); return 2; }
    fseek(fp, 0, SEEK_END); long size = ftell(fp); fseek(fp, 0, SEEK_SET);
    char* image = calloc(size + 1, 1);
    if (fread(image, 1, size, fp) != (size_t)size) { perror("read"); return 2; }
    fclose(fp);

    const char* libname = getenv("DRVRUN_LIBCUDA") ? getenv("DRVRUN_LIBCUDA") : "libcuda.so.1";
    lib = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen %s: %s\n", libname, dlerror()); return 2; }
#define LOAD(name) name = sym(#name)
    LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuDeviceGetName); LOAD(cuDeviceComputeCapability); LOAD(cuCtxCreate_v2);
    LOAD(cuCtxDestroy_v2); LOAD(cuCtxSynchronize); LOAD(cuModuleLoadDataEx); LOAD(cuModuleGetFunction);
    LOAD(cuMemAlloc_v2); LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoH_v2); LOAD(cuMemsetD8_v2); LOAD(cuLaunchKernel);
    LOAD(cuGetErrorName);

    CUdevice device; CUcontext context; char name[128]; int major, minor;
    CK(cuInit(0));
    CK(cuDeviceGet(&device, ordinal));
    CK(cuDeviceGetName(name, sizeof name, device));
    CK(cuDeviceComputeCapability(&major, &minor, device));
    printf("device %d: %s, sm_%d%d\n", ordinal, name, major, minor);
    CK(cuCtxCreate_v2(&context, 0, device));

    static char info[1 << 16], error[1 << 16];
    int options[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
                     CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* values[] = {info, (void*)(uintptr_t)sizeof info, error, (void*)(uintptr_t)sizeof error};
    CUmodule module;
    CUresult r = cuModuleLoadDataEx(&module, image, 4, options, values);
    if (r) {
        printf("FAIL cuModuleLoadDataEx -> %d %s\n%s\n", r, ename(r), error);
        return 1;
    }
    CUdeviceptr d_in, d_out, d_aux;
    CK(cuMemAlloc_v2(&d_in, THREADS * MAXREG * 4));
    CK(cuMemAlloc_v2(&d_out, THREADS * MAXREG * 4));
    CK(cuMemAlloc_v2(&d_aux, AUX_WORDS * 4));
    int failed = 0, ran = 0;
    for (size_t t = 0; t < sizeof tests / sizeof *tests; t++) {
        if (!wanted(tests[t].name)) continue;
        // -a: the proxy takes rows to be aligned (unless VF_ALIGNED_ROWS=0); a row that is not would fault.
        if (aligned_only && strchr(tests[t].in, 'u')) continue;
        int result = run(module, &tests[t], d_in, d_out, d_aux);
        if (result < 0) continue;  // not in this module
        ran++;
        failed += result;
    }
    printf("%d kernels, %d wrong\n", ran, failed);
    CK(cuCtxDestroy_v2(context));
    return failed != 0;
}
