// A function CUDA 12.9's library does not have: the forwarder returns the library's "not supported"
// status and says so once on standard error, with the [volta-forever] prefix; the other functions still work.
#include "common.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <nvrtc.h>

int main(void) {
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) FAIL("cublasCreate");
    cudaEmulationSpecialValuesSupport supported;
    cublasStatus_t s1 = cublasGetEmulationSpecialValuesSupport(h, &supported);  // new in cuBLAS 13
    cublasStatus_t s2 = cublasGetEmulationSpecialValuesSupport(h, &supported);  // said once
    if (s1 != CUBLAS_STATUS_NOT_SUPPORTED || s2 != CUBLAS_STATUS_NOT_SUPPORTED) FAIL("cuBLAS stub returned %d, %d", (int)s1, (int)s2);
    int version = 0;
    if (cublasGetVersion(h, &version) != CUBLAS_STATUS_SUCCESS) FAIL("cublasGetVersion after the stub");
    cublasDestroy(h);
    printf("cublasGetEmulationSpecialValuesSupport -> %d (CUBLAS_STATUS_NOT_SUPPORTED is %d), twice\n", (int)s1, (int)CUBLAS_STATUS_NOT_SUPPORTED);
    OK("cublas-stub");

    cusparseSpGEAMDescr_t descr;
    cusparseStatus_t s3 = cusparseSpGEAM_createDescr(&descr);                     // new in cuSPARSE 12.9
    if (s3 != CUSPARSE_STATUS_NOT_SUPPORTED) FAIL("cuSPARSE stub returned %d", (int)s3);
    printf("cusparseSpGEAM_createDescr -> %d (CUSPARSE_STATUS_NOT_SUPPORTED is %d)\n", (int)s3, (int)CUSPARSE_STATUS_NOT_SUPPORTED);
    OK("cusparse-stub");

    size_t size = 0;
    nvrtcResult s4 = nvrtcGetTileIRSize(NULL, &size);                             // new in NVRTC 13
    if (s4 != NVRTC_ERROR_INTERNAL_ERROR) FAIL("NVRTC stub returned %d", (int)s4);
    printf("nvrtcGetTileIRSize -> %d (NVRTC_ERROR_INTERNAL_ERROR)\n", (int)s4);
    OK("nvrtc-stub");
    return 0;
}
