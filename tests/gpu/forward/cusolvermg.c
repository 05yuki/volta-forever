// cuSOLVERMg: a handle on the one GPU there is, a device selection, and the matrix-descriptor calls
// (the multi-GPU factorisations need several GPUs).
#include "common.h"
#include <cusolverMg.h>

#define SOLVER(call) do { cusolverStatus_t r_ = (call); if (r_ != CUSOLVER_STATUS_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

int main(void) {
    cusolverMgHandle_t h;
    SOLVER(cusolverMgCreate(&h));
    int devices[1] = {0};
    SOLVER(cusolverMgDeviceSelect(h, 1, devices));
    cudaLibMgGrid_t grid;
    SOLVER(cusolverMgCreateDeviceGrid(&grid, 1, 1, devices, CUDALIBMG_GRID_MAPPING_COL_MAJOR));
    cudaLibMgMatrixDesc_t desc;
    SOLVER(cusolverMgCreateMatrixDesc(&desc, 256, 256, 256, 256, CUDA_R_64F, grid));
    SOLVER(cusolverMgDestroyMatrixDesc(desc));
    SOLVER(cusolverMgDestroyGrid(grid));
    SOLVER(cusolverMgDestroy(h));
    printf("cuSOLVERMg: handle, device selection, grid and matrix descriptor created and destroyed\n");
    OK("cusolvermg");
    return 0;
}
