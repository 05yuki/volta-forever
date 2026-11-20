// mma_scaled_gen2.c: generate reference vectors for block-scaled MMA on the RTX 5090.
// At least 3000 cases per form, in documented classes.
// usage: mma_scaled_gen2 MODULE OUTPUT_DIR
// File format: uint32 kernel_id, uint32 n_cases, then per case: in[32*12], out[32*4].
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int CUresult; typedef int CUdevice;
typedef struct CUctx_st* CUcontext; typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction; typedef unsigned long long CUdeviceptr;
enum{CU_JIT_INFO_LOG_BUFFER=3,CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES=4,
     CU_JIT_ERROR_LOG_BUFFER=5,CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES=6};
static void* lib;
#define SYM(r,n,...) static r(*n)(__VA_ARGS__)
SYM(CUresult,cuInit,unsigned); SYM(CUresult,cuDeviceGet,CUdevice*,int);
SYM(CUresult,cuDeviceGetName,char*,int,CUdevice);
SYM(CUresult,cuCtxCreate_v2,CUcontext*,unsigned,CUdevice);
SYM(CUresult,cuCtxDestroy_v2,CUcontext);
SYM(CUresult,cuCtxSynchronize,void);
SYM(CUresult,cuModuleLoadDataEx,CUmodule*,const void*,unsigned,int*,void**);
SYM(CUresult,cuModuleGetFunction,CUfunction*,CUmodule,const char*);
SYM(CUresult,cuMemAlloc_v2,CUdeviceptr*,size_t);
SYM(CUresult,cuMemcpyHtoD_v2,CUdeviceptr,const void*,size_t);
SYM(CUresult,cuMemcpyDtoH_v2,void*,CUdeviceptr,size_t);
SYM(CUresult,cuMemsetD8_v2,CUdeviceptr,unsigned char,size_t);
SYM(CUresult,cuLaunchKernel,CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
    unsigned,void*,void**,void**);
SYM(CUresult,cuGetErrorName,CUresult,const char**);
static void* sym(const char* n){void* p=dlsym(lib,n);if(!p){fprintf(stderr,"missing %s\n",n);exit(2);}return p;}
static const char* ename(CUresult r){const char*s="?";cuGetErrorName(r,&s);return s?s:"?";}
#define CK(c) do{CUresult r_=(c);if(r_){fprintf(stderr,"FAIL %s -> %d %s\n",#c,r_,ename(r_));exit(1);}}while(0)
#define THREADS 32
#define NIN 12
#define NOUT 4
static uint32_t h_in[THREADS * NIN], h_out[THREADS * NOUT];
static uint32_t rng_s;
static uint32_t rng(void) { rng_s = rng_s * 1664525u + 1013904223u; return rng_s; }
static void fill_ab_random(void) {
    for (int t = 0; t < THREADS; t++)
        for (int r = 0; r < 6; r++) h_in[t*NIN+r] = rng();
}
static void fill_ab_code(uint8_t code) {
    uint32_t v = 0; for (int i=0;i<8;i++) v |= ((uint32_t)(code&0xf))<<(4*i);
    for (int t = 0; t < THREADS; t++) {
        for (int r=0;r<4;r++) h_in[t*NIN+r] = v;
        for (int r=4;r<6;r++) h_in[t*NIN+r] = v;
    }
}
static void set_c_all(float c0, float c1, float c2, float c3) {
    uint32_t b0,b1,b2,b3;
    memcpy(&b0,&c0,4); memcpy(&b1,&c1,4); memcpy(&b2,&c2,4); memcpy(&b3,&c3,4);
    for (int t=0;t<THREADS;t++) { h_in[t*NIN+6]=b0; h_in[t*NIN+7]=b1; h_in[t*NIN+8]=b2; h_in[t*NIN+9]=b3; }
}
static void set_scales_uniform(uint8_t sb) {
    uint32_t v = (uint32_t)sb|((uint32_t)sb<<8)|((uint32_t)sb<<16)|((uint32_t)sb<<24);
    for (int t=0;t<THREADS;t++) { h_in[t*NIN+10]=v; h_in[t*NIN+11]=v; }
}
static void set_scales_random(void) {
    for (int t=0;t<THREADS;t++) { h_in[t*NIN+10]=rng(); h_in[t*NIN+11]=rng(); }
}
// Set only one block's A to nonzero, rest to zero. For 4X: block b uses threads (b&1)*2..(b&1)*2+1, reg pair b>>1.
// For 2X: block b uses reg pair b. For 1X: all.
static void set_one_block_a(int block, int n_blocks, uint8_t code) {
    uint32_t v = 0; for (int i=0;i<8;i++) v |= ((uint32_t)(code&0xf))<<(4*i);
    for (int t = 0; t < THREADS; t++) {
        for (int r=0;r<4;r++) h_in[t*NIN+r] = 0;
        int t_in_group = t & 3;
        if (n_blocks == 4) {
            int src_pair = block & 1;
            int reg_pair = block >> 1;
            if (t_in_group/2 == src_pair) {
                h_in[t*NIN + 2*reg_pair] = v;     // upper row
                h_in[t*NIN + 2*reg_pair+1] = v;   // lower row
            }
        } else if (n_blocks == 2) {
            h_in[t*NIN + 2*block] = v;
            h_in[t*NIN + 2*block+1] = v;
        } else { // 1 block
            for (int r=0;r<4;r++) h_in[t*NIN+r] = v;
        }
    }
    // B: same pattern
    for (int t = 0; t < THREADS; t++) {
        for (int r=4;r<6;r++) h_in[t*NIN+r] = 0;
        int t_in_group = t & 3;
        if (n_blocks == 4) {
            int reg_pair = block >> 1;
            int src_pair = block & 1;
            if (t_in_group/2 == src_pair) h_in[t*NIN + 4 + reg_pair] = v;
        } else if (n_blocks == 2) {
            h_in[t*NIN + 4 + block] = v;
        } else {
            for (int r=4;r<6;r++) h_in[t*NIN+r] = v;
        }
    }
}
static CUresult run_kernel(CUmodule mod, const char* name, CUdeviceptr di, CUdeviceptr do_) {
    CUfunction func; CUresult r = cuModuleGetFunction(&func, mod, name);
    if (r) return r;
    CK(cuMemcpyHtoD_v2(di, h_in, sizeof(h_in)));
    CK(cuMemsetD8_v2(do_, 0xee, sizeof(h_out)));
    void* args[] = {&di, &do_, &di};
    CK(cuLaunchKernel(func, 1,1,1, THREADS,1,1, 0, NULL, args, NULL));
    r = cuCtxSynchronize(); if (r) return r;
    CK(cuMemcpyDtoH_v2(h_out, do_, sizeof(h_out)));
    return 0;
}
typedef struct { const char* name; int kernel_id; int n_blocks; } Kernel;
static const Kernel kernels[] = {
    {"s_mxf4_2x", 0, 2},
    {"s_nvf4_4x", 1, 4},
    {"s_mxf8_e4m3_1x", 2, 1},
};
#define N_KERNELS 3
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mma_scaled_gen2 MODULE OUTPUT_DIR\n"); return 2; }
    FILE* fp = fopen(argv[1], "rb"); if (!fp) { perror(argv[1]); return 2; }
    fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
    char* img = calloc(sz+1,1); if(fread(img,1,sz,fp)!=(size_t)sz){perror("fread");return 2;} fclose(fp);
    lib = dlopen(getenv("DRVRUN_LIBCUDA") ? getenv("DRVRUN_LIBCUDA") : "libcuda.so.1", RTLD_NOW|RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
#define L(n) n=sym(#n)
    L(cuInit);L(cuDeviceGet);L(cuDeviceGetName);L(cuCtxCreate_v2);L(cuCtxDestroy_v2);
    L(cuCtxSynchronize);L(cuModuleLoadDataEx);L(cuModuleGetFunction);L(cuMemAlloc_v2);
    L(cuMemcpyHtoD_v2);L(cuMemcpyDtoH_v2);L(cuMemsetD8_v2);L(cuLaunchKernel);L(cuGetErrorName);
    CUdevice dev; CUcontext ctx; char name[128];
    CK(cuInit(0)); CK(cuDeviceGet(&dev,0)); CK(cuDeviceGetName(name,128,dev));
    printf("device: %s\n", name);
    CK(cuCtxCreate_v2(&ctx,0,dev));
    static char info_buf[65536], error_buf[65536];
    int opts[]={3,4,5,6}; void* vals[]={info_buf,(void*)65536ULL,error_buf,(void*)65536ULL};
    CUmodule mod;
    CUresult r = cuModuleLoadDataEx(&mod, img, 4, opts, vals);
    if (r) { fprintf(stderr, "FAIL load -> %d %s\n%s\n", r, ename(r), error_buf); return 1; }
    CUdeviceptr di, do_;
    CK(cuMemAlloc_v2(&di, sizeof(h_in)));
    CK(cuMemAlloc_v2(&do_, sizeof(h_out)));
    for (int k = 0; k < N_KERNELS; k++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s_large.bin", argv[2], kernels[k].name);
        FILE* out = fopen(path, "wb");
        if (!out) { perror(path); continue; }
        uint32_t n_cases = 0;
        fwrite(&kernels[k].kernel_id, 4, 1, out);
        fwrite(&n_cases, 4, 1, out);
        int nb = kernels[k].n_blocks;
        // Scale byte for 1.0: UE8M0 byte 127, UE4M3 byte 0x38.
        uint8_t scale_one = (k == 1) ? 0x38 : 127;
        // Scale byte for NaN: 0xff for UE8M0, 0x7f for UE4M3.
        uint8_t scale_nan = (k == 1) ? 0x7f : 0xff;
        // Scale byte for 0: 0 for both (UE8M0: 2^-127; UE4M3: 0).
        // Scale byte for max: UE8M0 254 (2^127), UE4M3 0x7e (448).
        uint8_t scale_max = (k == 1) ? 0x7e : 254;
        rng_s = 100 + k * 1000;
#define WC() do { \
    CUresult r_ = run_kernel(mod, kernels[k].name, di, do_); \
    if (r_) { fprintf(stderr, "%s case %d: FAIL %d %s\n", kernels[k].name, n_cases, r_, ename(r_)); break; } \
    fwrite(h_in, 4, THREADS*NIN, out); fwrite(h_out, 4, THREADS*NOUT, out); n_cases++; \
} while(0)
        // CLASS A: random operands, random scales (1500 cases). Seed documented.
        for (int i = 0; i < 1500; i++) {
            memset(h_in, 0, sizeof(h_in));
            fill_ab_random(); set_scales_random(); set_c_all(0,0,0,0);
            WC();
        }
        // CLASS B: random operands, scale byte sweep for scale_A (256 cases).
        for (int sb = 0; sb < 256; sb++) {
            memset(h_in, 0, sizeof(h_in));
            fill_ab_random();
            uint32_t sa = (uint32_t)sb|((uint32_t)sb<<8)|((uint32_t)sb<<16)|((uint32_t)sb<<24);
            uint32_t sb_v = (uint32_t)scale_one|((uint32_t)scale_one<<8)|((uint32_t)scale_one<<16)|((uint32_t)scale_one<<24);
            for (int t=0;t<THREADS;t++) { h_in[t*NIN+10]=sa; h_in[t*NIN+11]=sb_v; }
            set_c_all(0,0,0,0);
            WC();
        }
        // CLASS C: special scales (NaN, 0, max) with random operands (50 cases).
        uint8_t specials[] = {0, scale_nan, scale_max, 1, scale_one};
        for (int si = 0; si < 5; si++) for (int sj = 0; sj < 5; sj++) {
            memset(h_in, 0, sizeof(h_in));
            fill_ab_random();
            set_scales_uniform(specials[si]);
            // override scale_B
            uint32_t sb2 = (uint32_t)specials[sj]|((uint32_t)specials[sj]<<8)|((uint32_t)specials[sj]<<16)|((uint32_t)specials[sj]<<24);
            for (int t=0;t<THREADS;t++) h_in[t*NIN+11] = sb2;
            set_c_all(0,0,0,0);
            WC();
        }
        // CLASS D: accumulators of several magnitudes (200 cases).
        float c_vals[] = {0, 1.0f, -1.0f, 1e10f, -1e10f, 1e-10f, -1e-10f, 1e30f, -1e30f, 1e38f, -1e38f};
        for (int ci = 0; ci < 11; ci++) {
            for (int rep = 0; rep < 18; rep++) {
                memset(h_in, 0, sizeof(h_in));
                fill_ab_random(); set_scales_random();
                set_c_all(c_vals[ci], -c_vals[ci], c_vals[ci]*0.5f, -c_vals[ci]*0.5f);
                WC();
            }
        }
        // CLASS E: rounding discriminators -- cases where the exact sum lands near
        // a half-ULP of f32 (500 cases). With all 1.0 operands (block sum = 16 or 32)
        // and carefully chosen C to land the f64 sum at a midpoint.
        // For scale=1, block_sum=32 (2X, all 1.0) or 16 (4X): total unscaled = 64.
        // We set C so that C + 64 is at an f32 midpoint.
        // An f32 midpoint at binade [2^e, 2^(e+1)): ULP = 2^(e-23). Half-ULP = 2^(e-24).
        // C + 64 = M + 0.5*ULP for integer M in [2^e, 2^(e+1)).
        // So C = M + 0.5*ULP - 64. The f64 sum is then exactly M + 0.5*ULP.
        // Truncation: M. RNE: M or M+ULP depending on M's last bit.
        for (int rep = 0; rep < 500; rep++) {
            memset(h_in, 0, sizeof(h_in));
            fill_ab_code(2); // all 1.0
            set_scales_uniform(scale_one);
            // Pick a random exponent and mantissa
            int exp_bias = 127 + (rng() % 20) - 5; // exponent around 122..141
            uint32_t mant = rng() & 0x7fffff; // random mantissa
            // Build the f32 value M
            uint32_t m_bits = ((uint32_t)exp_bias << 23) | mant;
            float m_f; memcpy(&m_f, &m_bits, 4);
            // Half-ULP for this binade
            int exp = exp_bias - 127;
            double half_ulp = ldexp(1.0, exp - 24);
            // C = M + half_ulp - 64 (the unscaled sum with all-1.0 operands is 64)
            double c_exact = (double)m_f + half_ulp - 64.0;
            // Also try just above and just below
            float c_f;
            if (rep % 3 == 0) { c_f = (float)c_exact; }
            else if (rep % 3 == 1) { c_f = (float)(c_exact + half_ulp * 0.01); }
            else { c_f = (float)(c_exact - half_ulp * 0.01); }
            // Negative sign for half the cases
            if (rep % 2 == 1) c_f = -c_f;
            set_c_all(c_f, -c_f, c_f * 0.5f, 0);
            WC();
        }
        // CLASS F: one block nonzero, rest zero (n_blocks * 50 cases).
        for (int b = 0; b < nb; b++) {
            for (int rep = 0; rep < 50; rep++) {
                memset(h_in, 0, sizeof(h_in));
                uint8_t code = (rng() % 15) + 1; // nonzero E2M1 code
                if (k == 2) code = (rng() % 126) + 1; // nonzero FP8 code (not NaN 0x7f)
                set_one_block_a(b, nb, code);
                set_scales_random();
                set_c_all(0,0,0,0);
                WC();
            }
        }
        // CLASS G: more random with random scales and random C (500 cases).
        for (int rep = 0; rep < 500; rep++) {
            memset(h_in, 0, sizeof(h_in));
            fill_ab_random(); set_scales_random();
            float c0 = ldexpf((float)(int)(rng()%2000-1000), (int)(rng()%40)-15);
            set_c_all(c0, -c0, c0*0.1f, 0);
            WC();
        }
        fseek(out, 4, SEEK_SET);
        fwrite(&n_cases, 4, 1, out);
        fclose(out);
        printf("%s: %d cases written\n", kernels[k].name, n_cases);
    }
    CK(cuCtxDestroy_v2(ctx));
    printf("done\n");
    return 0;
}
