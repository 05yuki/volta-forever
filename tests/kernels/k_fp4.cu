// Kernels that execute the E2M1 (FP4) and UE8M0 conversion instructions the proxy rewrites for
// Volta (inline PTX), for tcrun.c. Same conventions as k_fp8.cu.
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_120 -O3 -ptx -o k_fp4.135.sm120.ptx k_fp4.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// ---- cvt.rn.f16x2.e2m1x2: decode two e2m1 nibbles to two halves -----------------------------------
// in: one byte (low 8 bits of a word); out: one b32 register (f16x2).
KERNEL(fp4_decode_e2m1) {
    unsigned x = in[GI] & 0xff;
    unsigned d;
    asm volatile("{\n\t.reg .b8 tmp;\n\tcvt.u8.u32 tmp, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}

// ---- cvt.rn.satfinite.e2m1x2.f32: encode two f32 to packed e2m1 pair ------------------------------
// in: two f32 (a, b); out: one word with the packed e2m1 pair in its low 8 bits.
KERNEL(fp4_encode_e2m1) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b8 tmp;\n\tcvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2;\n\tcvt.u32.u8 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// ---- cvt.rp.satfinite.ue8m0x2.f32: encode two f32 to packed ue8m0 pair (round toward +inf) --------
// in: two f32 (a, b); out: one word with the packed ue8m0 pair in its low 16 bits.
KERNEL(fp4_encode_ue8m0_rp) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rp.satfinite.ue8m0x2.f32 tmp, %1, %2;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// ---- cvt.rz.satfinite.ue8m0x2.f32: encode two f32 to packed ue8m0 pair (round toward zero) --------
KERNEL(fp4_encode_ue8m0_rz) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rz.satfinite.ue8m0x2.f32 tmp, %1, %2;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// cvt.rn.satfinite.ue8m0x2.f32 is listed in the PTX manual but ptxas 13.5 rejects it on sm_120a:
// only .rz and .rp are accepted for ue8m0x2 from f32 (the census uses rp). The layer handles rn
// anyway (the same code handles all three rounding modes) but there is no native instruction to
// test against.

// ---- cvt.rn.bf16x2.ue8m0x2: decode two ue8m0 bytes to two bf16 halves ----------------------------
// in: one b16 (low 16 bits of a word); out: one b32 register (bf16x2).
KERNEL(fp4_decode_ue8m0) {
    unsigned x = in[GI] & 0xffff;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.bf16x2.ue8m0x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}
