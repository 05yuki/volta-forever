// rt_probe: what a CUDA-runtime + cuBLAS application does on this GPU: runtime/driver versions,
// a kernel launch from the program's own fat binary, then cuBLAS SGEMM and half-precision GEMM.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void affine(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 2.0f * x[i] + 1.0f;
}

#define SHOW(call) do { cudaError_t e_ = (call); printf("%-34s -> %d %s\n", #call, (int)e_, cudaGetErrorName(e_)); } while (0)

int main() {
    int count = -1, rt = 0, drv = 0;
    SHOW(cudaGetDeviceCount(&count));
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    printf("runtime %d, driver API %d, devices %d\n", rt, drv, count);
    cudaDeviceProp prop;
    SHOW(cudaGetDeviceProperties(&prop, 0));
    printf("device 0: %s, sm_%d%d\n", prop.name, prop.major, prop.minor);

    const int n = 1 << 16;
    std::vector<float> h(n);
    for (int i = 0; i < n; i++) h[i] = (float)i;
    float* d = nullptr;
    SHOW(cudaMalloc(&d, n * sizeof(float)));
    SHOW(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    affine<<<n / 256, 256>>>(d, n);
    SHOW(cudaGetLastError());
    SHOW(cudaDeviceSynchronize());
    SHOW(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < n; i++) bad += h[i] != 2.0f * i + 1.0f;
    printf("own kernel: %ld/%d wrong\n", bad, n);

    cublasHandle_t handle;
    cublasStatus_t s = cublasCreate(&handle);
    printf("cublasCreate -> %d %s\n", (int)s, cublasGetStatusString(s));
    if (s == CUBLAS_STATUS_SUCCESS) {
        int ver = 0;
        cublasGetVersion(handle, &ver);
        printf("cuBLAS version %d\n", ver);
        const int m = 256;
        std::vector<float> a(m * m), b(m * m), c(m * m);
        for (int i = 0; i < m * m; i++) { a[i] = (float)(i % 7) - 3.0f; b[i] = (float)(i % 5) - 2.0f; }
        float *da, *db, *dc;
        cudaMalloc(&da, m * m * 4); cudaMalloc(&db, m * m * 4); cudaMalloc(&dc, m * m * 4);
        cudaMemcpy(da, a.data(), m * m * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), m * m * 4, cudaMemcpyHostToDevice);
        const float one = 1.0f, zero = 0.0f;
        s = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, da, m, db, m, &zero, dc, m);
        printf("cublasSgemm -> %d %s, sync %s\n", (int)s, cublasGetStatusString(s), cudaGetErrorName(cudaDeviceSynchronize()));
        cudaMemcpy(c.data(), dc, m * m * 4, cudaMemcpyDeviceToHost);
        bad = 0;
        for (int j = 0; j < m; j += 37)
            for (int i = 0; i < m; i += 29) {
                float ref = 0;
                for (int k = 0; k < m; k++) ref += a[i + k * m] * b[k + j * m];
                bad += c[i + j * m] != ref;
            }
        printf("sgemm check: %ld wrong\n", bad);

        // half-precision GEMM with float32 accumulation: the tensor-core path on Volta and later
        std::vector<__half> ha(m * m), hb(m * m), hc(m * m);
        for (int i = 0; i < m * m; i++) { ha[i] = __float2half(a[i]); hb[i] = __float2half(b[i]); }
        __half *dha, *dhb, *dhc;
        cudaMalloc(&dha, m * m * 2); cudaMalloc(&dhb, m * m * 2); cudaMalloc(&dhc, m * m * 2);
        cudaMemcpy(dha, ha.data(), m * m * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dhb, hb.data(), m * m * 2, cudaMemcpyHostToDevice);
        s = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, dha, CUDA_R_16F, m, dhb, CUDA_R_16F, m,
                         &zero, dhc, CUDA_R_16F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        printf("cublasGemmEx(half) -> %d %s, sync %s\n", (int)s, cublasGetStatusString(s),
               cudaGetErrorName(cudaDeviceSynchronize()));
        cudaMemcpy(hc.data(), dhc, m * m * 2, cudaMemcpyDeviceToHost);
        bad = 0;
        for (int j = 0; j < m; j += 37)
            for (int i = 0; i < m; i += 29) {
                float ref = 0;
                for (int k = 0; k < m; k++) ref += a[i + k * m] * b[k + j * m];
                bad += __half2float(hc[i + j * m]) != ref;
            }
        printf("half gemm check: %ld wrong\n", bad);
        cublasDestroy(handle);
    }
    return 0;
}
