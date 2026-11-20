// cuSOLVER: LU factorisation and solve (getrf, getrs) and Cholesky (potrf, potrs) of a 64x64 system in
// double precision, checked by the residual; then the dense API's generic Xgetrf.
#include "common.h"
#include <cusolverDn.h>

#define SOLVER(call) do { cusolverStatus_t r_ = (call); if (r_ != CUSOLVER_STATUS_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

enum { N = 64 };

static double residual(const double *a, const double *x, const double *b) {  // max |A x - b|, A column-major
    double worst = 0;
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int j = 0; j < N; j++) s += a[i + j * N] * x[j];
        worst = fmax(worst, fabs(s - b[i]));
    }
    return worst;
}

int main(void) {
    double *a = malloc(N * N * sizeof *a), *b = malloc(N * sizeof *b), *x = malloc(N * sizeof *x), *spd = malloc(N * N * sizeof *spd);
    for (int i = 0; i < N; i++) {
        b[i] = 1.0 + (i % 5);
        for (int j = 0; j < N; j++) a[i + j * N] = ((i * 31 + j * 17) % 23) / 23.0 + (i == j ? N : 0);
    }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {   // spd = A^T A + I
        double s = i == j;
        for (int k = 0; k < N; k++) s += a[k + i * N] * a[k + j * N];
        spd[i + j * N] = s;
    }
    cusolverDnHandle_t h;
    SOLVER(cusolverDnCreate(&h));
    double *dA = dev_alloc(N * N * sizeof *a), *dB = dev_alloc(N * sizeof *b), *work;
    int *piv = dev_alloc(N * sizeof(int)), *info = dev_alloc(sizeof(int)), hinfo = -1, lwork = 0;

    to_dev(dA, a, N * N * sizeof *a); to_dev(dB, b, N * sizeof *b);
    SOLVER(cusolverDnDgetrf_bufferSize(h, N, N, dA, N, &lwork));
    work = dev_alloc(lwork * sizeof *work);
    SOLVER(cusolverDnDgetrf(h, N, N, dA, N, work, piv, info));
    SOLVER(cusolverDnDgetrs(h, CUBLAS_OP_N, N, 1, dA, N, piv, dB, N, info));
    to_host(&hinfo, info, sizeof hinfo); to_host(x, dB, N * sizeof *x);
    if (hinfo != 0) FAIL("getrs info %d", hinfo);
    double r1 = residual(a, x, b);
    if (r1 > 1e-9) FAIL("LU residual %g", r1);

    to_dev(dA, spd, N * N * sizeof *a); to_dev(dB, b, N * sizeof *b);
    SOLVER(cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, N, dA, N, &lwork));
    work = dev_alloc(lwork * sizeof *work);
    SOLVER(cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, N, dA, N, work, lwork, info));
    SOLVER(cusolverDnDpotrs(h, CUBLAS_FILL_MODE_LOWER, N, 1, dA, N, dB, N, info));
    to_host(&hinfo, info, sizeof hinfo); to_host(x, dB, N * sizeof *x);
    if (hinfo != 0) FAIL("potrs info %d", hinfo);
    double r2 = residual(spd, x, b);
    if (r2 > 1e-8) FAIL("Cholesky residual %g", r2);

    // the generic 64-bit API
    to_dev(dA, a, N * N * sizeof *a); to_dev(dB, b, N * sizeof *b);
    cusolverDnParams_t params;
    SOLVER(cusolverDnCreateParams(&params));
    size_t dev_bytes = 0, host_bytes = 0;
    SOLVER(cusolverDnXgetrf_bufferSize(h, params, N, N, CUDA_R_64F, dA, N, CUDA_R_64F, &dev_bytes, &host_bytes));
    void *dwork = dev_alloc(dev_bytes ? dev_bytes : 1), *hwork = malloc(host_bytes ? host_bytes : 1);
    int64_t *piv64 = dev_alloc(N * sizeof(int64_t));
    SOLVER(cusolverDnXgetrf(h, params, N, N, CUDA_R_64F, dA, N, piv64, CUDA_R_64F, dwork, dev_bytes, hwork, host_bytes, info));
    SOLVER(cusolverDnXgetrs(h, params, CUBLAS_OP_N, N, 1, CUDA_R_64F, dA, N, piv64, CUDA_R_64F, dB, N, info));
    to_host(&hinfo, info, sizeof hinfo); to_host(x, dB, N * sizeof *x);
    double r3 = residual(a, x, b);
    if (hinfo != 0 || r3 > 1e-9) FAIL("Xgetrf info %d residual %g", hinfo, r3);
    int major = 0, minor = 0, patch = 0;
    SOLVER(cusolverGetProperty(MAJOR_VERSION, &major));
    SOLVER(cusolverGetProperty(MINOR_VERSION, &minor));
    SOLVER(cusolverGetProperty(PATCH_LEVEL, &patch));
    SOLVER(cusolverDnDestroy(h));
    printf("cuSOLVER version reported: %d.%d.%d; residuals: LU %g, Cholesky %g, Xgetrf %g\n", major, minor, patch, r1, r2, r3);
    OK("cusolver");
    return 0;
}
