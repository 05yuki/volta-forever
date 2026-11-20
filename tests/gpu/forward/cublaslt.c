// cuBLASLt: a single-precision matrix product through cublasLtMatmul, with the heuristic asked both ways
// (cublasLtMatmulAlgoGetHeuristic and the 13-only ...ForStream, which the forwarder answers with the
// first), checked against the host.
#include "common.h"
#include <cublasLt.h>

#define LT(call) do { cublasStatus_t r_ = (call); if (r_ != CUBLAS_STATUS_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

int main(void) {
    enum { M = 96, N = 64, K = 80 };
    float *a = malloc(M * K * sizeof(float)), *b = malloc(K * N * sizeof(float)), *c = malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) a[i] = (float)((i * 7) % 11) / 11.0f - 0.4f;
    for (int i = 0; i < K * N; i++) b[i] = (float)((i * 5) % 13) / 13.0f - 0.3f;
    float *dA = dev_alloc(M * K * sizeof(float)), *dB = dev_alloc(K * N * sizeof(float)), *dC = dev_alloc(M * N * sizeof(float));
    to_dev(dA, a, M * K * sizeof(float)); to_dev(dB, b, K * N * sizeof(float));
    cublasLtHandle_t h;
    LT(cublasLtCreate(&h));
    cublasLtMatmulDesc_t op;
    LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtMatrixLayout_t la, lb, lc;   // column-major
    LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, M, K, M));
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, N, K));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, M, N, M));
    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws_bytes = 1 << 22;
    void *ws = dev_alloc(ws_bytes);
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes));
    cublasLtMatmulHeuristicResult_t r1, r2;
    int n1 = 0, n2 = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 1, &r1, &n1));
    LT(cublasLtMatmulAlgoGetHeuristicForStream(h, op, la, lb, lc, lc, pref, 1, &r2, &n2, 0));
    if (n1 < 1 || n2 < 1) FAIL("no algorithm (%d, %d)", n1, n2);
    float alpha = 1.5f, beta = 0.0f;
    LT(cublasLtMatmul(h, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc, &r1.algo, ws, ws_bytes, 0));
    CUDA(cudaDeviceSynchronize());
    to_host(c, dC, M * N * sizeof(float));
    double worst = 0;
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        double s = 0;
        for (int k = 0; k < K; k++) s += (double)a[i + k * M] * b[k + j * K];
        worst = fmax(worst, fabs(alpha * s - c[i + j * M]));
    }
    if (worst > 1e-3) FAIL("product differs from the host by %g", worst);
    printf("cuBLASLt version reported: %zu; product max error %g; heuristics: %d and %d algorithms\n", cublasLtGetVersion(), worst, n1, n2);
    OK("cublaslt");
    return 0;
}
