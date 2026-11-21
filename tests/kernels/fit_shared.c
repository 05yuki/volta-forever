// Test the dynamic shared-memory opt-in at the real V100 limit for both driver APIs.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { \
    fprintf(stderr, "%s failed with %d\n", #call, (int)r_); return 1; } } while (0)
#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } } while (0)

static const char ptx[] =
    ".version 6.4\n.target sm_70\n.address_size 64\n"
    ".extern .shared .align 4 .b8 dynamic[];\n"
    ".visible .entry fill(.param .u64 output, .param .u32 words) {\n"
    ".shared .align 4 .b8 fixed[4096]; .reg .u32 r<10>; .reg .u64 a<4>; .reg .pred p;\n"
    "ld.param.u64 a0,[output]; ld.param.u32 r1,[words]; mov.u32 r0,%tid.x;\n"
    "mov.u32 r2,fixed; shl.b32 r3,r0,2; add.u32 r2,r2,r3; add.u32 r4,r0,1;\n"
    "st.volatile.shared.u32 [r2],r4; mov.u32 r5,dynamic; mov.u32 r6,r0;\n"
    "fill_loop: setp.ge.u32 p,r6,r1; @p bra filled; mad.lo.u32 r7,r6,4,r5;\n"
    "xor.b32 r8,r6,12345; st.volatile.shared.u32 [r7],r8; add.u32 r6,r6,1024; bra fill_loop;\n"
    "filled: bar.sync 0; ld.volatile.shared.u32 r4,[r2]; mad.wide.u32 a1,r0,4,a0;\n"
    "st.global.u32 [a1],r4; add.u64 a0,a0,4096; mov.u32 r6,r0;\n"
    "read_loop: setp.ge.u32 p,r6,r1; @p bra done; mad.lo.u32 r7,r6,4,r5;\n"
    "ld.volatile.shared.u32 r8,[r7]; mad.wide.u32 a1,r6,4,a0; st.global.u32 [a1],r8;\n"
    "add.u32 r6,r6,1024; bra read_loop; done: ret; }\n";

static int check_entry(int library, int enabled, CUdevice dev, int block_limit) {
    CUmodule module = NULL;
    CUlibrary lib = NULL;
    CUkernel kernel = NULL;
    CUfunction function;
    if (library) {
        CHECK(cuLibraryLoadData(&lib, ptx, NULL, NULL, 0, NULL, NULL, 0));
        CHECK(cuLibraryGetKernel(&kernel, lib, "fill"));
        CHECK(cuKernelGetFunction(&function, kernel));
    } else {
        CHECK(cuModuleLoadData(&module, ptx));
        CHECK(cuModuleGetFunction(&function, module, "fill"));
    }
    const CUfunction_attribute dynamic = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    const CUfunction_attribute fixed = CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;
    int static_bytes, previous, effective;
    CHECK(cuFuncGetAttribute(&static_bytes, fixed, function));
    REQUIRE(static_bytes == 4096);
    int max_bytes = block_limit - static_bytes;
    CHECK(cuFuncGetAttribute(&previous, dynamic, function));
    CUresult result = library ? cuKernelSetAttribute(dynamic, 101376, kernel, dev)
                              : cuFuncSetAttribute(function, dynamic, 101376);
    REQUIRE(result == (enabled ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE));
    if (library) {
        CHECK(cuKernelGetAttribute(&effective, dynamic, kernel, dev));
        REQUIRE(cuKernelSetAttribute(fixed, 101376, kernel, dev) == CUDA_ERROR_INVALID_VALUE);
    } else {
        CHECK(cuFuncGetAttribute(&effective, dynamic, function));
        REQUIRE(cuFuncSetAttribute(function, fixed, 101376) == CUDA_ERROR_INVALID_VALUE);
    }
    REQUIRE(effective == (enabled ? max_bytes : previous));
    if (!enabled) {
        if (library) CHECK(cuKernelSetAttribute(dynamic, max_bytes, kernel, dev));
        else CHECK(cuFuncSetAttribute(function, dynamic, max_bytes));
    }
    CUdeviceptr output;
    int words = max_bytes / 4;
    size_t bytes = static_bytes + max_bytes;
    unsigned *host = malloc(bytes);
    REQUIRE(host != NULL);
    CHECK(cuMemAlloc(&output, bytes));
    CHECK(cuMemsetD32(output, 0xFFFFFFFFu, bytes / 4));
    void *args[] = {&output, &words};
    CHECK(cuLaunchKernel(function, 1, 1, 1, 1024, 1, 1, max_bytes, NULL, args, NULL));
    CHECK(cuMemcpyDtoH(host, output, bytes));
    for (int i = 0; i < 1024; i++) REQUIRE(host[i] == (unsigned)i + 1);
    for (int i = 0; i < words; i++) REQUIRE(host[1024 + i] == ((unsigned)i ^ 12345u));
    result = cuLaunchKernel(function, 1, 1, 1, 1024, 1, 1, max_bytes + 4, NULL, args, NULL);
    REQUIRE(result == CUDA_ERROR_INVALID_VALUE || result == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    printf("%s: static %d, dynamic %d, %zu words verified, oversized launch refused\n",
           library ? "kernel" : "function", static_bytes, max_bytes, bytes / 4);
    free(host);
    CHECK(cuMemFree(output));
    if (library) CHECK(cuLibraryUnload(lib)); else CHECK(cuModuleUnload(module));
    return 0;
}

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    int limit;
    const char *setting = getenv("VF_FIT_SHARED");
    int enabled = setting && strcmp(setting, "1") == 0;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuDeviceGetAttribute(&limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    REQUIRE(limit == 98304);
    if (check_entry(0, enabled, dev, limit) || check_entry(1, enabled, dev, limit)) return 1;
    CHECK(cuDevicePrimaryCtxRelease(dev));
    puts("shared opt-in checks passed");
    return 0;
}
