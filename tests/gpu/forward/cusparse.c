// cuSPARSE: y = alpha * A x + beta * y with A in CSR form (generic API), and the same product as a
// sparse-by-dense matrix product (cusparseSpMM), against the host.
#include "common.h"
#include <cusparse.h>

#define SPARSE(call) do { cusparseStatus_t r_ = (call); if (r_ != CUSPARSE_STATUS_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

int main(void) {
    enum { N = 500, PER_ROW = 5, NNZ = N * PER_ROW, COLS = 3 };
    int *rows = malloc((N + 1) * sizeof(int)), *cols = malloc(NNZ * sizeof(int));
    float *vals = malloc(NNZ * sizeof(float)), *x = malloc(N * COLS * sizeof(float)), *y = malloc(N * COLS * sizeof(float)), *want = malloc(N * COLS * sizeof(float));
    for (int i = 0; i <= N; i++) rows[i] = i * PER_ROW;
    for (int i = 0; i < N; i++) for (int k = 0; k < PER_ROW; k++) {
        cols[i * PER_ROW + k] = (i * 37 + k * 101) % N;
        vals[i * PER_ROW + k] = 0.5f + (float)((i + 3 * k) % 7);
    }
    // columns inside a row must be sorted and distinct for a canonical CSR
    for (int i = 0; i < N; i++) for (int a = 0; a < PER_ROW; a++) for (int b = a + 1; b < PER_ROW; b++)
        if (cols[i * PER_ROW + b] < cols[i * PER_ROW + a]) {
            int c = cols[i * PER_ROW + a]; cols[i * PER_ROW + a] = cols[i * PER_ROW + b]; cols[i * PER_ROW + b] = c;
            float v = vals[i * PER_ROW + a]; vals[i * PER_ROW + a] = vals[i * PER_ROW + b]; vals[i * PER_ROW + b] = v;
        }
    for (int i = 0; i < N * COLS; i++) { x[i] = (float)((i * 13) % 17) / 17.0f; y[i] = 1.0f; }
    int *d_rows = dev_alloc((N + 1) * sizeof(int)), *d_cols = dev_alloc(NNZ * sizeof(int));
    float *d_vals = dev_alloc(NNZ * sizeof(float)), *d_x = dev_alloc(N * COLS * sizeof(float)), *d_y = dev_alloc(N * COLS * sizeof(float));
    to_dev(d_rows, rows, (N + 1) * sizeof(int)); to_dev(d_cols, cols, NNZ * sizeof(int)); to_dev(d_vals, vals, NNZ * sizeof(float));
    to_dev(d_x, x, N * COLS * sizeof(float)); to_dev(d_y, y, N * COLS * sizeof(float));

    cusparseHandle_t h;
    SPARSE(cusparseCreate(&h));
    cusparseSpMatDescr_t A;
    SPARSE(cusparseCreateCsr(&A, N, N, NNZ, d_rows, d_cols, d_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    float alpha = 2.0f, beta = 0.5f;
    // SpMV on the first column
    cusparseDnVecDescr_t vx, vy;
    SPARSE(cusparseCreateDnVec(&vx, N, d_x, CUDA_R_32F));
    SPARSE(cusparseCreateDnVec(&vy, N, d_y, CUDA_R_32F));
    size_t bytes = 0;
    SPARSE(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, vx, &beta, vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
    void *buf = dev_alloc(bytes ? bytes : 1);
    SPARSE(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, vx, &beta, vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    to_host(y, d_y, N * sizeof(float));
    double worst = 0;
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int k = rows[i]; k < rows[i + 1]; k++) s += (double)vals[k] * x[cols[k]];
        worst = fmax(worst, fabs(alpha * s + beta * 1.0 - y[i]));
    }
    if (worst > 1e-3) FAIL("SpMV differs from the host by %g", worst);
    // SpMM: C = A B, B of 3 columns, column-major
    cusparseDnMatDescr_t B, C;
    float zero = 0.0f, one = 1.0f;
    SPARSE(cusparseCreateDnMat(&B, N, COLS, N, d_x, CUDA_R_32F, CUSPARSE_ORDER_COL));
    SPARSE(cusparseCreateDnMat(&C, N, COLS, N, d_y, CUDA_R_32F, CUSPARSE_ORDER_COL));
    SPARSE(cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, B, &zero, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
    void *buf2 = dev_alloc(bytes ? bytes : 1);
    SPARSE(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, B, &zero, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf2));
    to_host(want, d_y, N * COLS * sizeof(float));
    double worst2 = 0;
    for (int c = 0; c < COLS; c++) for (int i = 0; i < N; i++) {
        double s = 0;
        for (int k = rows[i]; k < rows[i + 1]; k++) s += (double)vals[k] * x[c * N + cols[k]];
        worst2 = fmax(worst2, fabs(s - want[c * N + i]));
    }
    if (worst2 > 1e-3) FAIL("SpMM differs from the host by %g", worst2);
    int version = 0;
    SPARSE(cusparseGetVersion(h, &version));
    SPARSE(cusparseDestroy(h));
    printf("cuSPARSE version reported: %d; SpMV max error %g, SpMM max error %g\n", version, worst, worst2);
    OK("cusparse");
    return 0;
}
