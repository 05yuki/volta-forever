// Conversions of parameters of 8 and 16 bits to floats and doubles, which sm_75 code does from
// uniform registers (I2F.U8, I2F.S8, I2F.U16, I2F.S16, I2F.F64.U16 of a UR), for the differential
// test of the translator (diffrun.c).
//   toolkits/cuda-12.9.2/bin/nvcc -arch=sm_70 -O3 -cubin -o k_convert.sm70.cubin k_convert.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_convert.sm75.cubin k_convert.cu
#include <stdint.h>
#define I (blockIdx.x * blockDim.x + threadIdx.x)

extern "C" __global__ void k_uniform_convert(float* out, const float* x, int n, uint8_t a, int8_t b, uint16_t c, int16_t d,
                                             float scale, double bound, int rounds) {
    int i = I;
    if (i >= n) return;
    float acc = x[i];
    for (int r = 0; r < rounds; r++) {
        acc = acc * (float)a + (float)b;
        acc -= (float)c * 0.001f + (float)d;
        acc += (double)acc < bound ? 1.0f : 0.5f;
        acc = acc / scale;
        acc += (float)((int)scale + r);
    }
    out[i] = acc;
}

extern "C" __global__ void k_uniform_double(double* out, const double* x, int n, uint8_t a, uint16_t c, double m, double k, float f, int rounds) {
    int i = I;
    if (i >= n) return;
    double acc = x[i];
    for (int r = 0; r < rounds; r++) {
        acc = acc * m + k;
        acc += (double)a + (double)c;
        acc = acc > k ? acc - (double)f : acc;
        acc += (double)(long long)m;
    }
    out[i] = acc;
}
