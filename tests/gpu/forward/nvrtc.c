// NVRTC: compiles a kernel to machine code for the GPU (and to PTX), loads it with the driver API and
// runs it.
#include "common.h"
#include <cuda.h>
#include <nvrtc.h>

#define NVRTC(call) do { nvrtcResult r_ = (call); if (r_ != NVRTC_SUCCESS) FAIL("%s -> %d (%s)", #call, (int)r_, nvrtcGetErrorString(r_)); } while (0)
#define DRV(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

static const char *source =
    "extern \"C\" __global__ void axpy(float a, const float *x, float *y, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) y[i] = a * x[i] + y[i];\n"
    "}\n";

int main(void) {
    int major = 0, minor = 0;
    NVRTC(nvrtcVersion(&major, &minor));
    CUDA(cudaFree(0));
    int cc_major = 0, cc_minor = 0;
    CUDA(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
    CUDA(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));
    nvrtcProgram prog;
    NVRTC(nvrtcCreateProgram(&prog, source, "axpy.cu", 0, NULL, NULL));
    const char *options[] = {"--gpu-architecture=sm_70", "-default-device"};
    nvrtcResult r = nvrtcCompileProgram(prog, 2, options);
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    char *log = calloc(log_size + 1, 1);
    nvrtcGetProgramLog(prog, log);
    if (r != NVRTC_SUCCESS) FAIL("nvrtcCompileProgram -> %d: %s", (int)r, log);
    size_t cubin_size = 0;
    NVRTC(nvrtcGetCUBINSize(prog, &cubin_size));
    char *cubin = malloc(cubin_size);
    NVRTC(nvrtcGetCUBIN(prog, cubin));
    NVRTC(nvrtcDestroyProgram(&prog));

    CUmodule module;
    CUfunction kernel;
    DRV(cuModuleLoadData(&module, cubin));
    DRV(cuModuleGetFunction(&kernel, module, "axpy"));
    enum { N = 4096 };
    float *x = malloc(N * sizeof(float)), *y = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { x[i] = (float)i; y[i] = 1.0f; }
    float *dx = dev_alloc(N * sizeof(float)), *dy = dev_alloc(N * sizeof(float));
    to_dev(dx, x, N * sizeof(float)); to_dev(dy, y, N * sizeof(float));
    float a = 2.0f;
    int n = N;
    void *args[] = {&a, &dx, &dy, &n};
    DRV(cuLaunchKernel(kernel, N / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
    CUDA(cudaDeviceSynchronize());
    to_host(y, dy, N * sizeof(float));
    for (int i = 0; i < N; i++) if (y[i] != 2.0f * i + 1.0f) FAIL("y[%d] = %g", i, y[i]);

    // PTX, too: it names the architecture asked for.
    nvrtcProgram prog2;
    NVRTC(nvrtcCreateProgram(&prog2, source, "axpy.cu", 0, NULL, NULL));
    const char *ptx_options[] = {"--gpu-architecture=compute_70", "-default-device"};
    NVRTC(nvrtcCompileProgram(prog2, 2, ptx_options));
    size_t ptx_size = 0;
    NVRTC(nvrtcGetPTXSize(prog2, &ptx_size));
    char *ptx = malloc(ptx_size);
    NVRTC(nvrtcGetPTX(prog2, ptx));
    if (!strstr(ptx, ".target sm_70")) FAIL("the PTX has no `.target sm_70`");
    printf("NVRTC version reported: %d.%d; GPU is sm_%d%d; compiled for sm_70: %zu-byte cubin, ran: y = 2x + 1 correct; PTX targets sm_70\n", major, minor, cc_major, cc_minor, cubin_size);
    OK("nvrtc");
    return 0;
}
