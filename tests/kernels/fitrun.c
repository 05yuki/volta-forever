// fitrun: the test of the second translation at launch (docs/architecture.md, "Launch"), through the driver API alone.
//
// usage: fitrun [-d ORDINAL] [-x] [-s] [-t LAUNCHES] k_fit_test.135.sm75.cubin
//
// The kernels of k_fit_test.cu declare 64 registers; the layer's usual translation for sm_70 takes
// more, and the driver refuses a block of 1,024 threads of it (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES).
// The layer then launches a second translation of the module, made within the kernel's budget.
// Checked here, each with its results compared with the host's:
//   - a kernel taken from a module (cuModuleGetFunction) and one taken from a library
//     (cuLibraryGetKernel), with blocks of 256 threads (the first translation) and of 1,024 (the second);
//   - again from a module loaded in a second context;
//   - a kernel that needs 64 kB of dynamic shared memory: refused without the attribute that allows
//     it, launched once the program has set it on ITS kernel (cuFuncSetAttribute for a function,
//     cuKernelSetAttribute for a library's kernel, read back with cuKernelGetAttribute): the layer
//     must have given the attribute to the second translation's kernel too;
//   - a kernel that reads data of its module, after another kernel wrote it: right from the first
//     translation, and its blocks of 1,024 threads REFUSED whatever the layer can do (a second
//     translation is another module, where nobody wrote);
//   - a kernel of a module from which the program asked for a variable by name
//     (cuModuleGetGlobal_v2, what the headers of CUDA 13 call cuModuleGetGlobal): its blocks of
//     1,024 threads REFUSED too, for the same reason;
//   - the launch entry for the per-thread default stream (cuLaunchKernel_ptsz), a library's kernel
//     taken by enumeration (cuLibraryEnumerateKernels), and a function taken from the module that
//     the driver gives for a library (cuLibraryGetModule): launched from the second translation.
// -x: the control. The blocks of 1,024 threads are EXPECTED to be refused with 701 (run with
//     VF_FIT=0: without the mechanism the test must fail the other way).
// -t LAUNCHES: time that many launches of 1,024-thread blocks of the library's kernel (microseconds
//     each, the last one waited for). As it is and with VF_BORROW=5 (the first translation is then
//     the one within the budget: no refused call), the two times are the GPU's: the kernel's work.
//     With -x and VF_FIT=0 every launch is refused, and the time is that of the refused call alone.
//
// Only libcuda.so.1 is needed (dlopen; $DRVRUN_LIBCUDA names another one: the layer's).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUlib_st* CUlibrary;
typedef struct CUkern_st* CUkernel;
typedef unsigned long long CUdeviceptr;
enum { CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8, OUT_OF_RESOURCES = 701 };

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxDestroy_v2, CUcontext);
SYM(CUresult, cuCtxSetCurrent, CUcontext);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadData, CUmodule*, const void*);
SYM(CUresult, cuModuleUnload, CUmodule);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuFuncSetAttribute, CUfunction, int, int);
SYM(CUresult, cuLibraryLoadData, CUlibrary*, const void*, int*, void**, unsigned, int*, void**, unsigned);
SYM(CUresult, cuLibraryUnload, CUlibrary);
SYM(CUresult, cuLibraryGetKernel, CUkernel*, CUlibrary, const char*);
SYM(CUresult, cuKernelSetAttribute, int, int, CUkernel, CUdevice);
SYM(CUresult, cuKernelGetAttribute, int*, int, CUkernel, CUdevice);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemFree_v2, CUdeviceptr);
SYM(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuMemsetD32_v2, CUdeviceptr, unsigned, size_t);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    void*, void**, void**);
SYM(CUresult, cuLaunchKernel_ptsz, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    void*, void**, void**);
SYM(CUresult, cuModuleGetGlobal_v2, CUdeviceptr*, size_t*, CUmodule, const char*);
SYM(CUresult, cuLibraryGetKernelCount, unsigned*, CUlibrary);
SYM(CUresult, cuLibraryEnumerateKernels, CUkernel*, unsigned, CUlibrary);
SYM(CUresult, cuLibraryGetModule, CUmodule*, CUlibrary);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* n) { void* p = dlsym(lib, n); if (!p) { fprintf(stderr, "missing: %s\n", n); exit(2); } return p; }
static const char* ename(CUresult r) { const char* s = "?"; cuGetErrorName(r, &s); return s ? s : "?"; }
#define CHECK(call) do { CUresult r_ = (call); if (r_) { fprintf(stderr, "%s: %s (%d)\n", #call, ename(r_), r_); exit(1); } } while (0)

struct Shape { int kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw; };
enum { NACC = 40, N = 5000, WIDTH = 4096, WORDS = 16384 };
static const struct Shape shape = {2, 3, 4, 2, 1, 3, 1, 0, 2, 97, 13, 1};
static float input[WIDTH], expected[N];
static CUdeviceptr d_in, d_out;
static CUdevice device;
static int checks, wrong, expect_refusal;
static float scale = 1.0f, seen = 0.0f;         // what k_fit_writes_global was given
static int reads_data, written;                 // the kernel launched is k_fit_reads_global; the data was written
// What k_fit_reads_global adds to element i: fit_seen[i & 3], which is 0 as loaded.
static float added(int i) { return written ? seen + (float)(i & 3) : 0.0f; }

// What k_fit_test.cu's `walk` computes, on the host.
static float walk(int tid) {
    float a[NACC], sum = 0.0f;
    for (int k = 0; k < NACC; k++) a[k] = input[(tid + k) % WIDTH];
    for (int d = 0; d < shape.kd; d++)
        for (int h = 0; h < shape.kh; h++)
            for (int w = 0; w < shape.kw; w++) {
                int at = (d * shape.sd + shape.pd) * shape.dd + (h * shape.sh + shape.ph) * shape.dh + (w * shape.sw + shape.pw) * shape.dw;
                float x = input[(tid + at) % WIDTH];
                for (int k = 0; k < NACC; k++) a[k] = a[k] * 0.5f + x * (float)(k + 1);
            }
    for (int k = 0; k < NACC; k++) sum += a[k];
    return sum;
}

// One launch of `kernel` over N elements in blocks of `threads`, and what it left compared with the host's
// (`shared`: the kernel is k_fit_shared, with that many bytes of dynamic shared memory). `want`: 0 for a
// launch that must succeed, or the error it must be refused with (-1: any error).
static void run(const char* what, CUfunction kernel, int threads, unsigned shared, int want) {
    int n = N, width = WIDTH, words = WORDS;
    void* with_shared[] = {&d_out, &d_in, &n, &width, (void*)&shape, &words};
    void* plain[] = {&d_out, &d_in, &n, &width, (void*)&shape};
    unsigned blocks = (N + threads - 1) / threads;
    CHECK(cuMemsetD32_v2(d_out, 0x7fc00000u, N));          // (a NaN in every element: nothing left from another launch)
    CUresult r = cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, shared, NULL, shared ? with_shared : plain, NULL);
    checks++;
    if (want != 0) {
        int as_wanted = want == -1 ? r != 0 : r == want;
        if (!as_wanted) wrong++;
        printf("%s  %s, blocks of %d threads: %s (%d)%s\n", as_wanted ? "ok   " : "WRONG", what, threads, ename(r), r,
               as_wanted ? ", as expected" : r == 0 ? ": expected to be refused" : ": refused, but not with the expected error");
        return;
    }
    if (r != 0) {
        wrong++;
        printf("WRONG  %s, blocks of %d threads: %s (%d)\n", what, threads, ename(r), r);
        return;
    }
    CHECK(cuCtxSynchronize());
    static float got[N];
    CHECK(cuMemcpyDtoH_v2(got, d_out, sizeof got));
    int bad = 0, first = -1;
    for (int i = 0; i < N; i++) {
        float want_value = expected[i];
        if (shared) {
            // The neighbour in the block: the thread after this one, the first after the last. Its sum
            // is 0 when it is beyond N (the kernel computes nothing for it).
            int block = i / threads, t = i % threads, other = block * threads + (t + 1) % threads;
            want_value += other < N ? expected[other] : 0.0f;
        }
        if (reads_data) want_value = want_value * scale + added(i);
        if (!(fabsf(got[i] - want_value) <= 2e-4f * fabsf(want_value) + 1e-6f)) { if (first < 0) first = i; bad++; }
    }
    if (bad) {
        wrong++;
        printf("WRONG  %s, blocks of %d threads: %d of %d elements differ; the first, %d: %g, expected %g\n", what, threads, bad, N, first,
               got[first], reads_data ? expected[first] * scale + added(first) : expected[first]);
    } else {
        printf("ok     %s, blocks of %d threads: %d elements as on the host\n", what, threads, N);
    }
}

// The two block sizes: 256 threads fit the first translation; 1,024 need the second (or are refused, with -x).
static void both(const char* what, CUfunction kernel) {
    run(what, kernel, 256, 0, 0);
    run(what, kernel, 1024, 0, expect_refusal ? OUT_OF_RESOURCES : 0);
    run(what, kernel, 256, 0, 0);                        // (the first translation again, after the second was used)
    run(what, kernel, 1024, 0, expect_refusal ? OUT_OF_RESOURCES : 0);
}

static void buffers(void) {
    CHECK(cuMemAlloc_v2(&d_in, sizeof input));
    CHECK(cuMemAlloc_v2(&d_out, N * sizeof(float)));
    CHECK(cuMemcpyHtoD_v2(d_in, input, sizeof input));
}

int main(int argc, char** argv) {
    int requested_shared = 4 * WORDS, expected_shared = 4 * WORDS;
    int ordinal = 0, timed = 0, a = 1;
    for (; a < argc && argv[a][0] == '-'; a++) {
        if (!strcmp(argv[a], "-d") && a + 1 < argc) ordinal = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-t") && a + 1 < argc) timed = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-x")) expect_refusal = 1;
        else if (!strcmp(argv[a], "-s")) { requested_shared = 101376; expected_shared = 98304; }
        else { fprintf(stderr, "usage: fitrun [-d ORDINAL] [-x] [-s] [-t LAUNCHES] k_fit_test.135.sm75.cubin\n"); return 2; }
    }
    if (a + 1 != argc) { fprintf(stderr, "usage: fitrun [-d ORDINAL] [-x] [-s] [-t LAUNCHES] k_fit_test.135.sm75.cubin\n"); return 2; }
    FILE* f = fopen(argv[a], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[a]); return 2; }
    fseek(f, 0, SEEK_END); size_t size = ftell(f); fseek(f, 0, SEEK_SET);
    void* image = malloc(size);
    if (fread(image, 1, size, f) != size) { fprintf(stderr, "read error\n"); return 2; }
    fclose(f);

    const char* path = getenv("DRVRUN_LIBCUDA");
    lib = dlopen(path ? path : "libcuda.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "cannot open %s: %s\n", path ? path : "libcuda.so.1", dlerror()); return 2; }
#define LOAD(name) *(void**)&name = sym(#name)
    LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuCtxCreate_v2); LOAD(cuCtxDestroy_v2); LOAD(cuCtxSetCurrent); LOAD(cuCtxSynchronize);
    LOAD(cuModuleLoadData); LOAD(cuModuleUnload); LOAD(cuModuleGetFunction); LOAD(cuFuncSetAttribute);
    LOAD(cuLibraryLoadData); LOAD(cuLibraryUnload); LOAD(cuLibraryGetKernel); LOAD(cuKernelSetAttribute); LOAD(cuKernelGetAttribute);
    LOAD(cuMemAlloc_v2); LOAD(cuMemFree_v2); LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoH_v2); LOAD(cuMemsetD32_v2);
    LOAD(cuLaunchKernel); LOAD(cuLaunchKernel_ptsz);
    LOAD(cuModuleGetGlobal_v2); LOAD(cuLibraryGetKernelCount); LOAD(cuLibraryEnumerateKernels); LOAD(cuLibraryGetModule);
    LOAD(cuGetErrorName);

    unsigned seed = 12345;
    for (int i = 0; i < WIDTH; i++) { seed = seed * 1664525u + 1013904223u; input[i] = (float)(seed >> 8) / 16777216.0f - 0.5f; }
    for (int i = 0; i < N; i++) expected[i] = walk(i);

    CUcontext first, second;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&device, ordinal));
    CHECK(cuCtxCreate_v2(&first, 0, device));
    buffers();

    if (timed) {
        CUlibrary library; CUkernel kernel;
        CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
        CHECK(cuLibraryGetKernel(&kernel, library, "k_fit_regs"));
        int n = N, width = WIDTH;
        void* args[] = {&d_out, &d_in, &n, &width, (void*)&shape};
        for (int round = 0; round < 3; round++) {
            struct timespec t0, t1;
            CHECK(cuCtxSynchronize());
            clock_gettime(CLOCK_MONOTONIC, &t0);
            int refused = 0;
            for (int i = 0; i < timed; i++) {
                CUresult r = cuLaunchKernel((CUfunction)kernel, 5, 1, 1, 1024, 1, 1, 0, NULL, args, NULL);
                if (expect_refusal && r == OUT_OF_RESOURCES) refused++;
                else CHECK(r);
            }
            CHECK(cuCtxSynchronize());
            clock_gettime(CLOCK_MONOTONIC, &t1);
            printf("%d launches of 1,024-thread blocks%s: %.2f microseconds each\n", timed,
                   refused == timed ? ", every one refused" : refused ? ", SOME refused" : "",
                   ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1e3 / timed);
        }
        CHECK(cuLibraryUnload(library));
        CHECK(cuCtxDestroy_v2(first));
        return 0;
    }

    // ---- a module's function ----
    CUmodule module; CUfunction function;
    CHECK(cuModuleLoadData(&module, image));
    CHECK(cuModuleGetFunction(&function, module, "k_fit_regs"));
    both("function of a module", function);

    // ---- a library's kernel ----
    CUlibrary library; CUkernel kernel;
    CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
    CHECK(cuLibraryGetKernel(&kernel, library, "k_fit_regs"));
    both("kernel of a library", (CUfunction)kernel);
    // (Asked for again: the same kernel, and what the layer knows of it must not be lost.)
    CUkernel again;
    CHECK(cuLibraryGetKernel(&again, library, "k_fit_regs"));
    run("kernel of a library, asked for again", (CUfunction)again, 1024, 0, expect_refusal ? OUT_OF_RESOURCES : 0);

    // ---- dynamic shared memory that needs an attribute ----
    CUfunction shared_function; CUkernel shared_kernel;
    CHECK(cuModuleGetFunction(&shared_function, module, "k_fit_shared"));
    CHECK(cuLibraryGetKernel(&shared_kernel, library, "k_fit_shared"));
    run("64 kB of shared memory without the attribute, function of a module", shared_function, 1024, 4 * WORDS, -1);
    run("64 kB of shared memory without the attribute, kernel of a library", (CUfunction)shared_kernel, 1024, 4 * WORDS, -1);
    CHECK(cuFuncSetAttribute(shared_function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, requested_shared));
    CHECK(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, requested_shared, shared_kernel, device));
    int value = -1;
    CHECK(cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_kernel, device));
    checks++;
    if (value != expected_shared) { wrong++; printf("WRONG  the attribute read back from the library's kernel: %d, set to %d\n", value, expected_shared); }
    else printf("ok     the attribute read back from the library's kernel: %d\n", value);
    run("64 kB of shared memory with the attribute, function of a module", shared_function, 256, 4 * WORDS, 0);
    run("64 kB of shared memory with the attribute, function of a module", shared_function, 1024, 4 * WORDS, expect_refusal ? OUT_OF_RESOURCES : 0);
    run("64 kB of shared memory with the attribute, kernel of a library", (CUfunction)shared_kernel, 256, 4 * WORDS, 0);
    run("64 kB of shared memory with the attribute, kernel of a library", (CUfunction)shared_kernel, 1024, 4 * WORDS, expect_refusal ? OUT_OF_RESOURCES : 0);

    // ---- data of the module ----
    CUfunction writes, reads;
    CHECK(cuModuleGetFunction(&writes, module, "k_fit_writes_global"));
    CHECK(cuModuleGetFunction(&reads, module, "k_fit_reads_global"));
    reads_data = 1;
    run("kernel that reads data of its module, as loaded", reads, 256, 0, 0);
    scale = 3.0f; seen = 0.25f; written = 1;
    void* values[] = {&scale, &seen};
    CHECK(cuLaunchKernel(writes, 1, 1, 1, 32, 1, 1, 0, NULL, values, NULL));
    CHECK(cuCtxSynchronize());
    run("kernel that reads data of its module, after another kernel wrote it", reads, 256, 0, 0);
    run("kernel that reads data of its module (never from a second module)", reads, 1024, 0, OUT_OF_RESOURCES);
    run("kernel that reads data of its module, again", reads, 256, 0, 0);
    reads_data = 0;

    // ---- a second context: the library's kernel as it is, a module of its own ----
    CUdeviceptr in1 = d_in, out1 = d_out;
    CHECK(cuCtxCreate_v2(&second, 0, device));
    buffers();
    run("kernel of a library, in a second context", (CUfunction)kernel, 1024, 0, expect_refusal ? OUT_OF_RESOURCES : 0);
    CUmodule module2; CUfunction function2;
    CHECK(cuModuleLoadData(&module2, image));
    CHECK(cuModuleGetFunction(&function2, module2, "k_fit_regs"));
    both("function of a module loaded in a second context", function2);
    CHECK(cuModuleUnload(module2));
    CHECK(cuMemFree_v2(d_in)); CHECK(cuMemFree_v2(d_out));
    CHECK(cuCtxDestroy_v2(second));
    CHECK(cuCtxSetCurrent(first));
    d_in = in1; d_out = out1;
    run("function of a module, back in the first context", function, 1024, 0, expect_refusal ? OUT_OF_RESOURCES : 0);

    // ---- after an unload: the same image loaded again ----
    CHECK(cuModuleUnload(module));
    CHECK(cuLibraryUnload(library));
    CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
    CHECK(cuLibraryGetKernel(&kernel, library, "k_fit_regs"));
    run("kernel of a library loaded again after an unload", (CUfunction)kernel, 1024, 0, expect_refusal ? OUT_OF_RESOURCES : 0);
    CHECK(cuLibraryUnload(library));

    // A variable of the module asked for by name, by the name that the headers of CUDA 13 give the
    // call (cuModuleGetGlobal_v2): after that the module's kernels are not launched from a second
    // translation, which has its own copy of the variable. Blocks of 1,024 threads of k_fit_regs
    // stay refused, with or without the mechanism.
    CHECK(cuModuleLoadData(&module, image));
    CHECK(cuModuleGetFunction(&function, module, "k_fit_regs"));
    { CUdeviceptr dptr; size_t sz; CHECK(cuModuleGetGlobal_v2(&dptr, &sz, module, "fit_scale")); }
    run("function of a module with a variable asked for by name (cuModuleGetGlobal_v2)", function, 1024, 0, OUT_OF_RESOURCES);
    // Blocks of 256 threads run from the first translation as before.
    run("function of a module with a variable asked for by name (cuModuleGetGlobal_v2)", function, 256, 0, 0);
    CHECK(cuModuleUnload(module));

    // The launch entry for the per-thread default stream (cuLaunchKernel_ptsz): launched from the
    // second translation as by the plain entry.
    CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
    CHECK(cuLibraryGetKernel(&kernel, library, "k_fit_regs"));
    {
        int n = N, width = WIDTH;
        void* args[] = {&d_out, &d_in, &n, &width, (void*)&shape};
        unsigned blocks = (N + 1023) / 1024;
        CHECK(cuMemsetD32_v2(d_out, 0x7fc00000u, N));
        CUresult r = cuLaunchKernel_ptsz((CUfunction)kernel, blocks, 1, 1, 1024, 1, 1, 0, NULL, args, NULL);
        checks++;
        if (expect_refusal) {
            if (r != OUT_OF_RESOURCES) { wrong++; printf("WRONG  kernel of a library by cuLaunchKernel_ptsz, blocks of 1024 threads: %s (%d): expected to be refused\n", ename(r), r); }
            else printf("ok     kernel of a library by cuLaunchKernel_ptsz, blocks of 1024 threads: refused, as expected\n");
        } else {
            if (r != 0) { wrong++; printf("WRONG  kernel of a library by cuLaunchKernel_ptsz, blocks of 1024 threads: %s (%d)\n", ename(r), r); }
            else {
                CHECK(cuCtxSynchronize());
                static float got_ptsz[N];
                CHECK(cuMemcpyDtoH_v2(got_ptsz, d_out, sizeof got_ptsz));
                int bad = 0, first_bad = -1;
                for (int i = 0; i < N; i++)
                    if (!(fabsf(got_ptsz[i] - expected[i]) <= 2e-4f * fabsf(expected[i]) + 1e-6f))
                        { if (first_bad < 0) first_bad = i; bad++; }
                if (bad) { wrong++; printf("WRONG  kernel of a library by cuLaunchKernel_ptsz, blocks of 1024 threads: %d of %d elements differ (first at %d)\n", bad, N, first_bad); }
                else printf("ok     kernel of a library by cuLaunchKernel_ptsz, blocks of 1024 threads: %d elements as on the host\n", N);
            }
        }
    }
    CHECK(cuLibraryUnload(library));

    // A library's kernel taken by enumeration (cuLibraryEnumerateKernels), not by name.
    CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
    {
        unsigned kcount = 0;
        CHECK(cuLibraryGetKernelCount(&kcount, library));
        if (kcount > 0) {
            CUkernel *kbuf = calloc(kcount, sizeof(CUkernel));
            CHECK(cuLibraryEnumerateKernels(kbuf, kcount, library));
            // Find k_fit_regs among the enumerated kernels and launch it.
            CUkernel found = NULL;
            for (unsigned i = 0; i < kcount; i++) {
                const char *kn = NULL;
                // cuKernelGetName may not be available; use cuFuncGetName as fallback.
                void* fn_ptr = dlsym(lib, "cuKernelGetName");
                if (fn_ptr) {
                    CUresult nr = ((CUresult(*)(const char**, CUkernel))fn_ptr)(&kn, kbuf[i]);
                    if (nr != 0) kn = NULL;
                }
                if (!kn) {
                    void* fn_ptr2 = dlsym(lib, "cuFuncGetName");
                    if (fn_ptr2) ((CUresult(*)(const char**, CUfunction))fn_ptr2)(&kn, (CUfunction)kbuf[i]);
                }
                if (kn && !strcmp(kn, "k_fit_regs")) { found = kbuf[i]; break; }
            }
            if (found) {
                run("kernel of a library, taken by enumeration", (CUfunction)found, 1024, 0,
                    expect_refusal ? OUT_OF_RESOURCES : 0);
                run("kernel of a library, taken by enumeration", (CUfunction)found, 256, 0, 0);
            } else {
                checks++; wrong++;
                printf("WRONG  enumerated kernels: k_fit_regs not found among %u kernels\n", kcount);
            }
            free(kbuf);
        } else {
            checks++; wrong++;
            printf("WRONG  enumerated kernels: kernel count is 0\n");
        }
    }

    // A function taken from the module that the driver gives for the library (cuLibraryGetModule,
    // then cuModuleGetFunction).
    {
        CUmodule lib_module;
        CHECK(cuLibraryGetModule(&lib_module, library));
        CUfunction lib_func;
        CHECK(cuModuleGetFunction(&lib_func, lib_module, "k_fit_regs"));
        run("function of the module of a library", lib_func, 1024, 0,
            expect_refusal ? OUT_OF_RESOURCES : 0);
        run("function of the module of a library", lib_func, 256, 0, 0);
    }
    CHECK(cuLibraryUnload(library));

    // ---- enumeration with capacity larger than count: handles beyond the count untouched ----
    CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
    {
        unsigned kcount = 0;
        CHECK(cuLibraryGetKernelCount(&kcount, library));
        unsigned cap = kcount + 2;
        CUkernel *kbuf = calloc(cap, sizeof(CUkernel));
        // Sentinel values beyond the count.
        for (unsigned i = kcount; i < cap; i++) kbuf[i] = (CUkernel)(uintptr_t)0xdeadbeef;
        CHECK(cuLibraryEnumerateKernels(kbuf, cap, library));
        checks++;
        int sentinels_ok = 1;
        for (unsigned i = kcount; i < cap; i++)
            if (kbuf[i] != (CUkernel)(uintptr_t)0xdeadbeef) sentinels_ok = 0;
        if (!sentinels_ok) { wrong++; printf("WRONG  enumeration with excess capacity: sentinel handles changed\n"); }
        else printf("ok     enumeration with excess capacity: sentinel handles untouched\n");
        // The kernel from the first slot is usable.
        CUkernel found = NULL;
        for (unsigned i = 0; i < kcount; i++) {
            const char *kn = NULL;
            void* fn_ptr = dlsym(lib, "cuKernelGetName");
            if (fn_ptr) {
                CUresult nr = ((CUresult(*)(const char**, CUkernel))fn_ptr)(&kn, kbuf[i]);
                if (nr != 0) kn = NULL;
            }
            if (!kn) {
                void* fn_ptr2 = dlsym(lib, "cuFuncGetName");
                if (fn_ptr2) ((CUresult(*)(const char**, CUfunction))fn_ptr2)(&kn, (CUfunction)kbuf[i]);
            }
            if (kn && !strcmp(kn, "k_fit_regs")) { found = kbuf[i]; break; }
        }
        if (found)
            run("enumerated kernel with excess capacity", (CUfunction)found, 1024, 0,
                expect_refusal ? OUT_OF_RESOURCES : 0);
        else { checks++; wrong++; printf("WRONG  enumerated kernel with excess capacity: k_fit_regs not found\n"); }
        free(kbuf);
    }
    CHECK(cuLibraryUnload(library));

    // ---- attribute on the library's kernel follows to a function of the library's module ----
    // The kernel k_fit_shared needs CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES raised to use
    // 60000 bytes of dynamic shared memory at 1024 threads. Setting the attribute on K (CUkernel)
    // must make a launch through F (CUfunction from cuModuleGetFunction on the library's module)
    // succeed after a second translation.
    CHECK(cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0));
    {
        CUkernel shared_k;
        CHECK(cuLibraryGetKernel(&shared_k, library, "k_fit_shared"));
        CHECK(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, requested_shared, shared_k, device));
        CUmodule lib_module;
        CHECK(cuLibraryGetModule(&lib_module, library));
        CUfunction shared_f;
        CHECK(cuModuleGetFunction(&shared_f, lib_module, "k_fit_shared"));
        run("64 kB shared, attribute on K, launch through F of the module",
            shared_f, 1024, 4 * WORDS, expect_refusal ? OUT_OF_RESOURCES : 0);
    }
    CHECK(cuLibraryUnload(library));

    CHECK(cuMemFree_v2(d_in)); CHECK(cuMemFree_v2(d_out));
    CHECK(cuCtxDestroy_v2(first));
    printf("%d checks, %d wrong\n", checks, wrong);
    return wrong ? 1 : 0;
}
