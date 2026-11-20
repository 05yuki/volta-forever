// k_fp4_vectors.cu: one kernel for each conversion of E2M1 and UE8M0 that the compilers take,
// for fp4_vectors.c. Those with _relu in their name have the .relu modifier.
//
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_120a -O3 -ptx -o k_fp4_vectors.135.sm120a.ptx k_fp4_vectors.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out, const unsigned* aux)

// ---- E2M1 ----
KERNEL(truth_decode_e2m1) {
    unsigned x = in[GI] & 0xff;
    unsigned d;
    asm volatile("{\n\t.reg .b8 tmp;\n\tcvt.u8.u32 tmp, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(truth_decode_e2m1_relu) {
    unsigned x = in[GI] & 0xff;
    unsigned d;
    asm volatile("{\n\t.reg .b8 tmp;\n\tcvt.u8.u32 tmp, %1;\n\tcvt.rn.relu.f16x2.e2m1x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(truth_encode_e2m1) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b8 tmp;\n\tcvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2;\n\tcvt.u32.u8 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

KERNEL(truth_encode_e2m1_relu) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b8 tmp;\n\tcvt.rn.relu.satfinite.e2m1x2.f32 tmp, %1, %2;\n\tcvt.u32.u8 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// ---- UE8M0 from f32 ----
KERNEL(truth_encode_ue8m0_rp) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rp.satfinite.ue8m0x2.f32 tmp, %1, %2;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

KERNEL(truth_encode_ue8m0_rz) {
    float a = __uint_as_float(in[GI * 2]);
    float b = __uint_as_float(in[GI * 2 + 1]);
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rz.satfinite.ue8m0x2.f32 tmp, %1, %2;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "f"(a), "f"(b));
    out[GI] = d;
}

// ---- UE8M0 to bf16 ----
KERNEL(truth_decode_ue8m0) {
    unsigned x = in[GI] & 0xffff;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.bf16x2.ue8m0x2 %0, tmp;\n\t}" : "=r"(d) : "r"(x));
    out[GI] = d;
}

// ---- UE8M0 from bf16 ----
KERNEL(truth_encode_ue8m0_bf16_rp) {
    unsigned x = in[GI];  // bf16x2 packed
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rp.satfinite.ue8m0x2.bf16x2 tmp, %1;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(truth_encode_ue8m0_bf16_rz) {
    unsigned x = in[GI];  // bf16x2 packed
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.rz.satfinite.ue8m0x2.bf16x2 tmp, %1;\n\tcvt.u32.u16 %0, tmp;\n\t}"
                 : "=r"(d) : "r"(x));
    out[GI] = d;
}
