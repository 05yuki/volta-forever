// mma_scaled_refvec.c: generate reference vectors for block-scaled MMA on the RTX 5090.
//
// usage: mma_scaled_refvec MODULE OUTPUT_DIR
//
// For each kernel: sets up input registers for one warp (32 threads), runs the kernel,
// writes the input and output to a binary file. The inputs are designed to exercise:
// 1. All-ones E2M1 (code 2 = 1.0) with scale 1 (UE8M0 byte 127)
// 2. Each E2M1 value (0..15) in a fixed position, rest zero, scale 1
// 3. Random E2M1 values with varying scales
// 4. Scale byte 0 (2^-127) and byte 0xFF (NaN)
// 5. Nonzero C accumulator
//
// File format: a header (uint32_t kernel_id, uint32_t n_cases), then for each case:
//   uint32_t in[32*12], uint32_t out[32*4] (all thread registers in order).
//
// Environment: DRVRUN_LIBCUDA (library to load, default libcuda.so.1).
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

static void set_all_zero(void) { memset(h_in, 0, sizeof(h_in)); }

// Set scale registers to UE8M0 byte b for all threads.
static void set_scales_ue8m0(uint8_t b) {
    uint32_t v = (uint32_t)b | ((uint32_t)b << 8) | ((uint32_t)b << 16) | ((uint32_t)b << 24);
    for (int t = 0; t < THREADS; t++) {
        h_in[t * NIN + 10] = v;  // sa
        h_in[t * NIN + 11] = v;  // sb
    }
}

// Fill A and B registers with the given E2M1 nibble code (same code everywhere).
static void fill_ab_e2m1(uint8_t code) {
    uint32_t packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((uint32_t)(code & 0xf)) << (4 * i);
    for (int t = 0; t < THREADS; t++) {
        for (int r = 0; r < 4; r++) h_in[t * NIN + r] = packed;      // a0-a3
        for (int r = 4; r < 6; r++) h_in[t * NIN + r] = packed;      // b0-b1
    }
}

// Fill A and B with random E2M1 nibbles.
static void fill_ab_random(uint32_t seed) {
    uint32_t x = seed;
    for (int t = 0; t < THREADS; t++) {
        for (int r = 0; r < 6; r++) {
            x = x * 1664525u + 1013904223u;
            h_in[t * NIN + r] = x;  // random nibbles in a0-a3, b0-b1
        }
    }
}

// Set C accumulators.
static void set_c(float c0, float c1, float c2, float c3) {
    for (int t = 0; t < THREADS; t++) {
        uint32_t b0, b1, b2, b3;
        memcpy(&b0, &c0, 4); memcpy(&b1, &c1, 4); memcpy(&b2, &c2, 4); memcpy(&b3, &c3, 4);
        h_in[t * NIN + 6] = b0;
        h_in[t * NIN + 7] = b1;
        h_in[t * NIN + 8] = b2;
        h_in[t * NIN + 9] = b3;
    }
}

static CUresult run_kernel(CUmodule mod, const char* name, CUdeviceptr di, CUdeviceptr do_) {
    CUfunction func;
    CUresult r = cuModuleGetFunction(&func, mod, name);
    if (r) return r;
    CK(cuMemcpyHtoD_v2(di, h_in, sizeof(h_in)));
    CK(cuMemsetD8_v2(do_, 0xee, sizeof(h_out)));
    void* args[] = {&di, &do_};
    CK(cuLaunchKernel(func, 1, 1, 1, THREADS, 1, 1, 0, NULL, args, NULL));
    r = cuCtxSynchronize();
    if (r) return r;
    CK(cuMemcpyDtoH_v2(h_out, do_, sizeof(h_out)));
    return 0;
}

typedef struct { const char* name; int kernel_id; } Kernel;
static const Kernel kernels[] = {
    {"s_mxf4_2x", 0},
    {"s_nvf4_4x", 1},
    {"s_mxf8_e4m3_1x", 2},
};
#define N_KERNELS 3

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mma_scaled_refvec MODULE OUTPUT_DIR\n"); return 2; }
    FILE* fp = fopen(argv[1], "rb"); if (!fp) { perror(argv[1]); return 2; }
    fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
    char* img = calloc(sz+1,1); fread(img,1,sz,fp); fclose(fp);

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

    static char info[65536], error[65536];
    int opts[]={3,4,5,6}; void* vals[]={info,(void*)65536ULL,error,(void*)65536ULL};
    CUmodule mod;
    CUresult r = cuModuleLoadDataEx(&mod, img, 4, opts, vals);
    if (r) { fprintf(stderr, "FAIL load -> %d %s\n%s\n", r, ename(r), error); return 1; }

    CUdeviceptr di, do_;
    CK(cuMemAlloc_v2(&di, sizeof(h_in)));
    CK(cuMemAlloc_v2(&do_, sizeof(h_out)));

    for (int k = 0; k < N_KERNELS; k++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.bin", argv[2], kernels[k].name);
        FILE* out = fopen(path, "wb");
        if (!out) { perror(path); continue; }

        // Case 0: all ones (code 2 = 1.0), scale = 1 (byte 127), C = 0
        set_all_zero();
        fill_ab_e2m1(2); // 1.0
        set_scales_ue8m0(127); // 2^0 = 1
        set_c(0, 0, 0, 0);
        uint32_t n_cases = 0;
        fwrite(&kernels[k].kernel_id, 4, 1, out);
        fwrite(&n_cases, 4, 1, out); // placeholder, overwritten at the end

        #define WRITE_CASE() do { \
            CUresult r_ = run_kernel(mod, kernels[k].name, di, do_); \
            if (r_) { fprintf(stderr, "%s case %d: launch FAILED %d %s\n", kernels[k].name, n_cases, r_, ename(r_)); break; } \
            fwrite(h_in, 4, THREADS * NIN, out); \
            fwrite(h_out, 4, THREADS * NOUT, out); \
            n_cases++; \
        } while(0)

        WRITE_CASE(); // case 0: all 1.0, scale 1, C=0

        // Case 1: all 1.0, scale 2 (byte 128)
        set_all_zero();
        fill_ab_e2m1(2);
        set_scales_ue8m0(128);
        set_c(0, 0, 0, 0);
        WRITE_CASE();

        // Case 2: code 7 (E2M1 value 6), scale 1
        set_all_zero();
        fill_ab_e2m1(7); // 6.0
        set_scales_ue8m0(127);
        set_c(0, 0, 0, 0);
        WRITE_CASE();

        // Case 3: random values, scale 1
        set_all_zero();
        fill_ab_random(42 + k);
        set_scales_ue8m0(127);
        set_c(0, 0, 0, 0);
        WRITE_CASE();

        // Case 4: random values, scale 4 (byte 129)
        set_all_zero();
        fill_ab_random(123 + k);
        set_scales_ue8m0(129);
        set_c(0, 0, 0, 0);
        WRITE_CASE();

        // Case 5: all 1.0, scale 1, C = 42.0
        set_all_zero();
        fill_ab_e2m1(2);
        set_scales_ue8m0(127);
        set_c(42.0f, -1.0f, 100.0f, 0.5f);
        WRITE_CASE();

        // Case 6: scale byte 0 (2^-127)
        set_all_zero();
        fill_ab_e2m1(2);
        set_scales_ue8m0(0);
        set_c(0, 0, 0, 0);
        WRITE_CASE();

        // Case 7: random values, random scales
        set_all_zero();
        fill_ab_random(999 + k);
        {
            uint32_t x = 777 + k;
            for (int t = 0; t < THREADS; t++) {
                x = x * 1664525u + 1013904223u;
                h_in[t * NIN + 10] = x;
                x = x * 1664525u + 1013904223u;
                h_in[t * NIN + 11] = x;
            }
        }
        set_c(0, 0, 0, 0);
        WRITE_CASE();

        // Fix up case count.
        fseek(out, 4, SEEK_SET);
        fwrite(&n_cases, 4, 1, out);
        fclose(out);
        printf("%s: %d cases\n", kernels[k].name, n_cases);
    }

    CK(cuCtxDestroy_v2(ctx));
    return 0;
}
