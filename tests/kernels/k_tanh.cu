// tanhf compiled for sm_75 becomes the PTX instruction tanh.approx.f32, which Volta lacks.
#include <math.h>
extern "C" __global__ void tanhk(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}
