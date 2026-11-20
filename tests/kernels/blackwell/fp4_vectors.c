// fp4_vectors.c: the conversions of E2M1 and UE8M0 on fixed inputs, results written to files.
//
// usage: fp4_vectors [-d GPU] MODULE OUTPUT_DIR
//
// MODULE is k_fp4_vectors.135.sm120a.ptx (with `.version 9.0` for the driver of an RTX 5090, which
// takes no later one). For each kernel the module has, the inputs are made here, always the same:
// every value of the small formats, and for floats every representable value, every midpoint
// between two with the float on each side of it, zeros, infinities, NaN and the ends of the range.
// OUTPUT_DIR gets one file a kernel (inputs and results, or results alone where the inputs are
// 0, 1, 2, ...) and report.txt, which also says how many results differ from a model written
// here.
//
// ref/ holds the files an RTX 5090 wrote (2026-11-17, driver 580.178.04). The layer on a V100
// computes the same when its files are those, byte for byte: `cmp` (tests/gpu/run.sh, case
// fp4-hardware).
//
// Environment: DRVRUN_LIBCUDA (library to load, default libcuda.so.1).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
enum { CU_JIT_INFO_LOG_BUFFER=3,CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES=4,
       CU_JIT_ERROR_LOG_BUFFER=5,CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES=6 };
static void* lib;
#define SYM(ret,nm,...) static ret(*nm)(__VA_ARGS__)
SYM(CUresult,cuInit,unsigned); SYM(CUresult,cuDeviceGet,CUdevice*,int);
SYM(CUresult,cuDeviceGetName,char*,int,CUdevice);
SYM(CUresult,cuCtxCreate_v2,CUcontext*,unsigned,CUdevice);
SYM(CUresult,cuCtxDestroy_v2,CUcontext);
SYM(CUresult,cuCtxSynchronize,void);
SYM(CUresult,cuModuleLoadDataEx,CUmodule*,const void*,unsigned,int*,void**);
SYM(CUresult,cuModuleGetFunction,CUfunction*,CUmodule,const char*);
SYM(CUresult,cuMemAlloc_v2,CUdeviceptr*,size_t);
SYM(CUresult,cuMemFree_v2,CUdeviceptr);
SYM(CUresult,cuMemcpyHtoD_v2,CUdeviceptr,const void*,size_t);
SYM(CUresult,cuMemcpyDtoH_v2,void*,CUdeviceptr,size_t);
SYM(CUresult,cuMemsetD8_v2,CUdeviceptr,unsigned char,size_t);
SYM(CUresult,cuLaunchKernel,CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
    unsigned,void*,void**,void**);
SYM(CUresult,cuGetErrorName,CUresult,const char**);
static void* sym(const char* n){void* p=dlsym(lib,n);if(!p){fprintf(stderr,"missing %s\n",n);exit(2);}return p;}
static const char* ename(CUresult r){const char*s="?";cuGetErrorName(r,&s);return s?s:"?";}
#define CK(c) do{CUresult r_=(c);if(r_){fprintf(stderr,"FAIL %s -> %d %s\n",#c,r_,ename(r_));exit(1);}}while(0)

#define MAX_N 65536
#define BLK 256
static uint32_t h_in[MAX_N*2], h_out[MAX_N];

static double f2d(uint32_t b){float f;memcpy(&f,&b,4);return f;}
static uint32_t d2f(double v){float f=(float)v;uint32_t b;memcpy(&b,&f,4);return b;}
static uint32_t d2h(double v){
    uint32_t s=signbit(v)?0x8000:0; v=fabs(v);
    if(isnan(v))return 0x7e00; if(v>=65520.0)return s|0x7c00;
    if(v<ldexp(1,-14))return s|(uint32_t)nearbyint(ldexp(v,24));
    int e;double m=frexp(v,&e);uint32_t b=(uint32_t)nearbyint(ldexp(m,11));
    if(b==2048){b=1024;e++;}return s|((uint32_t)(e+14)<<10)|(b-1024);
}

// E2M1 value table
static double e2m1_val(uint8_t c){
    static const double t[16]={0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6};
    return t[c&0xf];
}
static uint8_t e2m1_nearest(double v){
    if(isnan(v))return 7;
    uint8_t s=signbit(v)?8:0; double a=fabs(v);
    if(a>=6.0)return s|7;
    uint8_t best=0;double bd=a;
    for(uint8_t c=1;c<8;c++){double d=fabs(a-e2m1_val(c));if(d<bd||(d==bd&&(c&1)==0)){best=c;bd=d;}}
    return s|best;
}
// UE8M0 value: 2^(e-127) for all e 0..254; NaN for 0xff.
static double ue8m0_val(uint8_t e){return e==0xff?NAN:ldexp(1.0,(int)e-127);}
// UE8M0 encode: the representable set is {2^(-127), 2^(-126), ..., 2^(127), NaN}.
// Byte 0 = 2^(-127). rz = floor, rp = ceil among powers of two. Zero -> byte 0, NaN -> 0xff,
// inf -> 0xfe (satfinite). The sign is ignored (unsigned format).
static uint8_t ue8m0_encode_f32(double v, int rounding){
    if(isnan(v))return 0xff;
    double a=fabs(v);
    if(a==0)return 0;
    if(isinf(a))return 0xfe;
    int e;
    double m=frexp(a,&e); // a = m * 2^e, 0.5 <= m < 1
    int actual_exp=e-1;    // a in [2^actual_exp, 2^(actual_exp+1))
    int is_exact=(m==0.5); // a == 2^actual_exp exactly
    int fb=actual_exp+127; // floor byte
    int cb=is_exact?fb:fb+1; // ceil byte
    if(fb<0)fb=0; if(cb<0)cb=0;
    if(fb>254)fb=254; if(cb>254)cb=254;
    return rounding==1?fb:cb;
}

// UE8M0 decode to bf16: 2^(e-127). e=0 → bf16 subnormal 0x0040. e=0xff → bf16 0x7fff.
static uint32_t ue8m0_to_bf16(uint8_t e){
    if(e==0xff) return 0x7fff;  // NaN with all mantissa bits
    if(e==0) return 0x0040;     // 2^(-127) as bf16 subnormal
    return (uint32_t)e<<7;      // normal: exponent field = e
}

// bf16 value
static double bf16_val(uint32_t bits){return f2d((bits&0xffff)<<16);}

// UE8M0 encode from bf16
static uint8_t ue8m0_encode_bf16(double v, int rounding){
    // Same logic but for bf16 input (which is an exact f32)
    return ue8m0_encode_f32(v, rounding);
}

static void launch(CUfunction func, int n, int words_per_in, CUdeviceptr di, CUdeviceptr do_, CUdeviceptr da){
    CK(cuMemcpyHtoD_v2(di, h_in, n*words_per_in*4));
    CK(cuMemsetD8_v2(do_, 0xee, n*4));
    void*args[]={&di,&do_,&da};
    CK(cuLaunchKernel(func,(n+BLK-1)/BLK,1,1,BLK,1,1,0,NULL,args,NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH_v2(h_out, do_, n*4));
}

// Generate UE8M0 encode inputs: exhaustive coverage of all 255 exponent ranges + subnormals + specials.
static int gen_ue8m0_encode_inputs(uint32_t* buf, int max){
    int n=0;
    // Exact zero, negative zero
    if(n+2<=max){buf[n++]=0x00000000;buf[n++]=0x00000000;}
    if(n+2<=max){buf[n++]=0x80000000;buf[n++]=0x80000000;}
    // All f32 subnormals (sample: smallest, mid, largest, and their negatives)
    static const uint32_t subns[]={0x00000001,0x00400000,0x007fffff};
    for(int i=0;i<3;i++){
        if(n+2<=max){buf[n++]=subns[i];buf[n++]=subns[i];}
        if(n+2<=max){buf[n++]=subns[i]|0x80000000;buf[n++]=subns[i]|0x80000000;}
    }
    // Every exact power of 2 from 2^(-126) to 2^(127)
    for(int e=-126;e<=127&&n+2<=max;e++){
        uint32_t b=d2f(ldexp(1.0,e));
        buf[n++]=b;buf[n++]=b;
    }
    // Halfway between consecutive powers: 2^e * 1.5
    for(int e=-126;e<=126&&n+2<=max;e++){
        uint32_t b=d2f(ldexp(1.5,e));
        buf[n++]=b;buf[n++]=b;
        // One ULP below and above the midpoint
        if(n+2<=max){buf[n++]=b-1;buf[n++]=b-1;}
        if(n+2<=max){buf[n++]=b+1;buf[n++]=b+1;}
    }
    // Just above and below each power of 2
    for(int e=-126;e<=127&&n+2<=max;e++){
        uint32_t b=d2f(ldexp(1.0,e));
        if(n+2<=max){buf[n++]=b+1;buf[n++]=b+1;}
        if(b>0&&n+2<=max){buf[n++]=b-1;buf[n++]=b-1;}
    }
    // Negatives of some powers
    for(int e=-126;e<=127&&n+2<=max;e+=5){
        uint32_t b=d2f(-ldexp(1.0,e));
        buf[n++]=b;buf[n++]=b;
        b=d2f(-ldexp(1.5,e));
        if(n+2<=max){buf[n++]=b;buf[n++]=b;}
    }
    // Inf, NaN variants
    static const uint32_t sp[]={0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0x7fffffff,0xff7fffff};
    for(int i=0;i<7&&n+2<=max;i++){buf[n++]=sp[i];buf[n++]=sp[i];}
    return n;
}

int main(int argc, char** argv){
    int gpu=0;
    int a=1;
    for(;a<argc&&argv[a][0]=='-';a++){
        if(!strcmp(argv[a],"-d"))gpu=atoi(argv[++a]);
    }
    if(a+1>=argc){fprintf(stderr,"usage: fp4_vectors [-d GPU] MODULE OUTPUT_DIR\n");return 2;}
    const char*modpath=argv[a], *outdir=argv[a+1];
    FILE*fp=fopen(modpath,"rb");if(!fp){perror(modpath);return 2;}
    fseek(fp,0,SEEK_END);long sz=ftell(fp);fseek(fp,0,SEEK_SET);
    char*image=calloc(sz+1,1);
    if(fread(image,1,sz,fp)!=(size_t)sz){fprintf(stderr,"%s: cut short\n",modpath);return 2;}
    fclose(fp);

    lib=dlopen(getenv("DRVRUN_LIBCUDA")?getenv("DRVRUN_LIBCUDA"):"libcuda.so.1",RTLD_NOW|RTLD_GLOBAL);
    if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());return 2;}
#define L(n) n=sym(#n)
    L(cuInit);L(cuDeviceGet);L(cuDeviceGetName);L(cuCtxCreate_v2);L(cuCtxDestroy_v2);
    L(cuCtxSynchronize);L(cuModuleLoadDataEx);L(cuModuleGetFunction);L(cuMemAlloc_v2);
    L(cuMemFree_v2);L(cuMemcpyHtoD_v2);L(cuMemcpyDtoH_v2);L(cuMemsetD8_v2);
    L(cuLaunchKernel);L(cuGetErrorName);

    CUdevice dev;CUcontext ctx;char name[128];
    CK(cuInit(0));CK(cuDeviceGet(&dev,gpu));CK(cuDeviceGetName(name,128,dev));
    printf("device %d: %s\n",gpu,name);
    CK(cuCtxCreate_v2(&ctx,0,dev));

    static char info[1<<16],error[1<<16];
    int opts[]={CU_JIT_INFO_LOG_BUFFER,CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,CU_JIT_ERROR_LOG_BUFFER,CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void*vals[]={info,(void*)(uintptr_t)sizeof info,error,(void*)(uintptr_t)sizeof error};
    CUmodule mod;
    CUresult r=cuModuleLoadDataEx(&mod,image,4,opts,vals);
    if(r){fprintf(stderr,"FAIL load -> %d %s\n%s\n",r,ename(r),error);return 1;}

    CUdeviceptr di,do_,da;
    CK(cuMemAlloc_v2(&di,MAX_N*2*4));
    CK(cuMemAlloc_v2(&do_,MAX_N*4));
    CK(cuMemAlloc_v2(&da,4096));

    char path[1024]; FILE*out;
    snprintf(path,sizeof path,"%s/report.txt",outdir);
    out=fopen(path,"w");if(!out){perror(path);return 2;}
    fprintf(out,"Truth extraction on %s\n\n",name);
    int total_wrong=0;

    // ---- Decode E2M1: all 256 bytes ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_decode_e2m1")){
            for(int i=0;i<256;i++)h_in[i]=i;
            launch(func,256,1,di,do_,da);
            snprintf(path,sizeof path,"%s/decode_e2m1.bin",outdir);
            fp=fopen(path,"wb");fwrite(h_out,4,256,fp);fclose(fp);
            int w=0;
            for(int i=0;i<256;i++){
                uint32_t exp=d2h(e2m1_val(i&0xf))|(uint32_t)d2h(e2m1_val((i>>4)&0xf))<<16;
                if(h_out[i]!=exp){if(w<10)fprintf(out,"  decode_e2m1[0x%02x]: hw=%08x model=%08x\n",i,h_out[i],exp);w++;}
            }
            fprintf(out,"decode_e2m1: %d/256 wrong\n",w);printf("decode_e2m1: %d/256 wrong\n",w);total_wrong+=w;
        }else printf("truth_decode_e2m1: not found\n");
    }

    // ---- Decode E2M1 with .relu: all 256 bytes ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_decode_e2m1_relu")){
            for(int i=0;i<256;i++)h_in[i]=i;
            launch(func,256,1,di,do_,da);
            snprintf(path,sizeof path,"%s/decode_e2m1_relu.bin",outdir);
            fp=fopen(path,"wb");fwrite(h_out,4,256,fp);fclose(fp);
            // .relu: negative values become +0
            int w=0;
            for(int i=0;i<256;i++){
                double lo_v=e2m1_val(i&0xf), hi_v=e2m1_val((i>>4)&0xf);
                if(lo_v<0||signbit(lo_v))lo_v=0.0;
                if(hi_v<0||signbit(hi_v))hi_v=0.0;
                uint32_t exp=d2h(lo_v)|(uint32_t)d2h(hi_v)<<16;
                if(h_out[i]!=exp){if(w<10)fprintf(out,"  decode_e2m1_relu[0x%02x]: hw=%08x model=%08x\n",i,h_out[i],exp);w++;}
            }
            fprintf(out,"decode_e2m1_relu: %d/256 wrong\n",w);printf("decode_e2m1_relu: %d/256 wrong\n",w);total_wrong+=w;
        }else printf("truth_decode_e2m1_relu: not found\n");
    }

    // ---- Encode E2M1: boundary values ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_encode_e2m1")){
            // Generate thorough inputs
            int n=0;
            static const double vals[]={0,0.5,1,1.5,2,3,4,6};
            static const double bounds[]={0.25,0.75,1.25,1.75,2.5,3.5,5.0};
            // Every exact value and negative
            for(int i=0;i<8&&n+2<=MAX_N*2;i++){h_in[n++]=d2f(vals[i]);h_in[n++]=d2f(-vals[i]);}
            for(int i=0;i<8&&n+2<=MAX_N*2;i++){h_in[n++]=d2f(-vals[i]);h_in[n++]=d2f(vals[i]);}
            // Boundaries and neighbors
            for(int i=0;i<7;i++)for(int s=0;s<2;s++){
                double b=s?-bounds[i]:bounds[i];uint32_t bits=d2f(b);
                if(n+2<=MAX_N*2){h_in[n++]=bits;h_in[n++]=bits;}
                if(n+2<=MAX_N*2){h_in[n++]=bits-1;h_in[n++]=bits-1;}
                if(n+2<=MAX_N*2){h_in[n++]=bits+1;h_in[n++]=bits+1;}
            }
            // Specials: 0, -0, inf, -inf, NaN, large, denorm
            static const uint32_t sp[]={0x00000000,0x80000000,0x7f800000,0xff800000,
                0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fbfffff,0xffbfffff,
                0x7fffffff,0xffffffff,0x41c80000,0x42480000,0x477fe000,0x00000001,
                0x80000001,0x00800000,0x80800000};
            for(int i=0;i<19&&n+2<=MAX_N*2;i++){h_in[n++]=sp[i];h_in[n++]=sp[i];}
            // Various exponents
            for(int e=-126;e<=127&&n+2<=MAX_N*2;e++){uint32_t b=d2f(ldexp(1.0,e));h_in[n++]=b;h_in[n++]=d2f(-ldexp(1.0,e));}
            for(int e=-126;e<=127&&n+2<=MAX_N*2;e++){uint32_t b=d2f(ldexp(1.5,e));h_in[n++]=b;h_in[n++]=d2f(-ldexp(1.5,e));}
            int threads=n/2;
            launch(func,threads,2,di,do_,da);
            snprintf(path,sizeof path,"%s/encode_e2m1.bin",outdir);
            fp=fopen(path,"wb");
            for(int i=0;i<threads;i++){fwrite(&h_in[i*2],4,1,fp);fwrite(&h_in[i*2+1],4,1,fp);fwrite(&h_out[i],4,1,fp);}
            fclose(fp);
            int w=0;
            for(int i=0;i<threads;i++){
                uint8_t up=e2m1_nearest(f2d(h_in[i*2])),lo=e2m1_nearest(f2d(h_in[i*2+1]));
                uint32_t exp=(uint32_t)(up<<4|lo);
                if((h_out[i]&0xff)!=(exp&0xff)){
                    if(w<10)fprintf(out,"  encode_e2m1: a=%08x b=%08x hw=%02x model=%02x\n",
                        h_in[i*2],h_in[i*2+1],h_out[i]&0xff,exp&0xff);w++;}
            }
            fprintf(out,"encode_e2m1: %d/%d wrong\n",w,threads);printf("encode_e2m1: %d/%d wrong\n",w,threads);total_wrong+=w;
        }
    }

    // ---- Encode E2M1 with .relu ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_encode_e2m1_relu")){
            // Reuse inputs from above, but test: .relu clamps negative results to +0
            int n=0;
            // (The last 0 was read past the end of 23 values when the reference file was written on
            // the RTX 5090, and was a zero there: it is kept, so that the inputs are the file's.)
            static const double test_vals[]={-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6,
                -0.24,-0.26,0.24,0.26,-7,7,-100,100,0};
            for(int i=0;i<24&&n+2<=MAX_N*2;i++){h_in[n++]=d2f(test_vals[i]);h_in[n++]=d2f(test_vals[i]);}
            // Also -0, NaN, inf, -inf
            static const uint32_t sp2[]={0x80000000,0x7fc00000,0xffc00000,0x7f800000,0xff800000};
            for(int i=0;i<5&&n+2<=MAX_N*2;i++){h_in[n++]=sp2[i];h_in[n++]=sp2[i];}
            int threads=n/2;
            launch(func,threads,2,di,do_,da);
            snprintf(path,sizeof path,"%s/encode_e2m1_relu.bin",outdir);
            fp=fopen(path,"wb");
            for(int i=0;i<threads;i++){fwrite(&h_in[i*2],4,1,fp);fwrite(&h_in[i*2+1],4,1,fp);fwrite(&h_out[i],4,1,fp);}
            fclose(fp);
            // Model: .relu makes negative inputs encode as +0 (nibble 0)
            int w=0;
            for(int i=0;i<threads;i++){
                double va=f2d(h_in[i*2]),vb=f2d(h_in[i*2+1]);
                // .relu: clamp input to >= 0 before rounding? Or clamp result?
                // Try: clamp input, then round
                uint8_t up,lo;
                if(isnan(va)) up=7; else if(va<=0) up=0; else up=e2m1_nearest(va);
                if(isnan(vb)) lo=7; else if(vb<=0) lo=0; else lo=e2m1_nearest(vb);
                uint32_t exp=(uint32_t)(up<<4|lo);
                if((h_out[i]&0xff)!=(exp&0xff)){
                    if(w<10)fprintf(out,"  encode_e2m1_relu: a=%08x(%.4g) b=%08x(%.4g) hw=%02x model=%02x\n",
                        h_in[i*2],va,h_in[i*2+1],vb,h_out[i]&0xff,exp&0xff);w++;}
            }
            fprintf(out,"encode_e2m1_relu: %d/%d wrong\n",w,threads);printf("encode_e2m1_relu: %d/%d wrong\n",w,threads);total_wrong+=w;
        }else printf("truth_encode_e2m1_relu: not found\n");
    }

    // ---- Decode UE8M0: all 65536 ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_decode_ue8m0")){
            for(int i=0;i<65536;i++)h_in[i]=i;
            launch(func,65536,1,di,do_,da);
            snprintf(path,sizeof path,"%s/decode_ue8m0.bin",outdir);
            fp=fopen(path,"wb");fwrite(h_out,4,65536,fp);fclose(fp);
            int w=0;
            for(int i=0;i<65536;i++){
                uint32_t exp=ue8m0_to_bf16(i&0xff)|(ue8m0_to_bf16((i>>8)&0xff)<<16);
                if(h_out[i]!=exp){if(w<5)fprintf(out,"  decode_ue8m0[%04x]: hw=%08x model=%08x\n",i,h_out[i],exp);w++;}
            }
            fprintf(out,"decode_ue8m0: %d/65536 wrong\n",w);printf("decode_ue8m0: %d/65536 wrong\n",w);total_wrong+=w;
        }
    }

    // ---- Encode UE8M0 rp from f32 ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_encode_ue8m0_rp")){
            int n=gen_ue8m0_encode_inputs(h_in,MAX_N*2);int threads=n/2;
            launch(func,threads,2,di,do_,da);
            snprintf(path,sizeof path,"%s/encode_ue8m0_rp.bin",outdir);
            fp=fopen(path,"wb");
            for(int i=0;i<threads;i++){fwrite(&h_in[i*2],4,1,fp);fwrite(&h_in[i*2+1],4,1,fp);fwrite(&h_out[i],4,1,fp);}
            fclose(fp);
            int w=0;
            for(int i=0;i<threads;i++){
                uint8_t up=ue8m0_encode_f32(f2d(h_in[i*2]),2),lo=ue8m0_encode_f32(f2d(h_in[i*2+1]),2);
                uint32_t exp=(uint32_t)(up<<8|lo);
                if((h_out[i]&0xffff)!=(exp&0xffff)){
                    if(w<10)fprintf(out,"  encode_ue8m0_rp: a=%08x(%.6g) b=%08x(%.6g) hw=%04x model=%04x\n",
                        h_in[i*2],f2d(h_in[i*2]),h_in[i*2+1],f2d(h_in[i*2+1]),h_out[i]&0xffff,exp&0xffff);w++;}
            }
            fprintf(out,"encode_ue8m0_rp: %d/%d wrong\n",w,threads);printf("encode_ue8m0_rp: %d/%d wrong\n",w,threads);total_wrong+=w;
        }
    }

    // ---- Encode UE8M0 rz from f32 ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_encode_ue8m0_rz")){
            int n=gen_ue8m0_encode_inputs(h_in,MAX_N*2);int threads=n/2;
            launch(func,threads,2,di,do_,da);
            snprintf(path,sizeof path,"%s/encode_ue8m0_rz.bin",outdir);
            fp=fopen(path,"wb");
            for(int i=0;i<threads;i++){fwrite(&h_in[i*2],4,1,fp);fwrite(&h_in[i*2+1],4,1,fp);fwrite(&h_out[i],4,1,fp);}
            fclose(fp);
            int w=0;
            for(int i=0;i<threads;i++){
                uint8_t up=ue8m0_encode_f32(f2d(h_in[i*2]),1),lo=ue8m0_encode_f32(f2d(h_in[i*2+1]),1);
                uint32_t exp=(uint32_t)(up<<8|lo);
                if((h_out[i]&0xffff)!=(exp&0xffff)){
                    if(w<10)fprintf(out,"  encode_ue8m0_rz: a=%08x(%.6g) b=%08x(%.6g) hw=%04x model=%04x\n",
                        h_in[i*2],f2d(h_in[i*2]),h_in[i*2+1],f2d(h_in[i*2+1]),h_out[i]&0xffff,exp&0xffff);w++;}
            }
            fprintf(out,"encode_ue8m0_rz: %d/%d wrong\n",w,threads);printf("encode_ue8m0_rz: %d/%d wrong\n",w,threads);total_wrong+=w;
        }
    }

    // ---- Encode UE8M0 from bf16 rp ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_encode_ue8m0_bf16_rp")){
            // All 65536 bf16x2 patterns would be huge, so test: every bf16 value paired with 0x3f80 (=1.0)
            int n=0;
            for(int i=0;i<65536&&n<MAX_N;i++){
                h_in[n++]=i|(0x3f80<<16); // lo=bf16 pattern i, hi=1.0
            }
            launch(func,n,1,di,do_,da);
            snprintf(path,sizeof path,"%s/encode_ue8m0_bf16_rp.bin",outdir);
            fp=fopen(path,"wb");
            for(int i=0;i<n;i++){fwrite(&h_in[i],4,1,fp);fwrite(&h_out[i],4,1,fp);}
            fclose(fp);
            int w=0;
            for(int i=0;i<n;i++){
                double lo_v=bf16_val(h_in[i]&0xffff), hi_v=bf16_val(h_in[i]>>16);
                uint8_t up=ue8m0_encode_f32(hi_v,2),lo=ue8m0_encode_f32(lo_v,2);
                uint32_t exp=(uint32_t)(up<<8|lo);
                if((h_out[i]&0xffff)!=(exp&0xffff)){
                    if(w<10)fprintf(out,"  encode_ue8m0_bf16_rp[%04x]: hw=%04x model=%04x (%.6g,%.6g)\n",
                        h_in[i],h_out[i]&0xffff,exp&0xffff,lo_v,hi_v);w++;}
            }
            fprintf(out,"encode_ue8m0_bf16_rp: %d/%d wrong\n",w,n);printf("encode_ue8m0_bf16_rp: %d/%d wrong\n",w,n);total_wrong+=w;
        }else printf("truth_encode_ue8m0_bf16_rp: not found\n");
    }

    // ---- Encode UE8M0 from bf16 rz ----
    {
        CUfunction func;
        if(!cuModuleGetFunction(&func,mod,"truth_encode_ue8m0_bf16_rz")){
            int n=0;
            for(int i=0;i<65536&&n<MAX_N;i++){
                h_in[n++]=i|(0x3f80<<16);
            }
            launch(func,n,1,di,do_,da);
            snprintf(path,sizeof path,"%s/encode_ue8m0_bf16_rz.bin",outdir);
            fp=fopen(path,"wb");
            for(int i=0;i<n;i++){fwrite(&h_in[i],4,1,fp);fwrite(&h_out[i],4,1,fp);}
            fclose(fp);
            int w=0;
            for(int i=0;i<n;i++){
                double lo_v=bf16_val(h_in[i]&0xffff), hi_v=bf16_val(h_in[i]>>16);
                uint8_t up=ue8m0_encode_f32(hi_v,1),lo=ue8m0_encode_f32(lo_v,1);
                uint32_t exp=(uint32_t)(up<<8|lo);
                if((h_out[i]&0xffff)!=(exp&0xffff)){
                    if(w<10)fprintf(out,"  encode_ue8m0_bf16_rz[%04x]: hw=%04x model=%04x (%.6g,%.6g)\n",
                        h_in[i],h_out[i]&0xffff,exp&0xffff,lo_v,hi_v);w++;}
            }
            fprintf(out,"encode_ue8m0_bf16_rz: %d/%d wrong\n",w,n);printf("encode_ue8m0_bf16_rz: %d/%d wrong\n",w,n);total_wrong+=w;
        }else printf("truth_encode_ue8m0_bf16_rz: not found\n");
    }

    fprintf(out,"\nTotal wrong across all forms: %d\n",total_wrong);
    fclose(out);
    printf("\nTotal wrong: %d\n",total_wrong);
    CK(cuCtxDestroy_v2(ctx));
    return total_wrong!=0;
}
