// cuRAND: uniform and normal numbers from the device generator, checked by their statistics, and the
// same seed twice giving the same numbers.
#include "common.h"
#include <curand.h>

#define CURAND(call) do { curandStatus_t r_ = (call); if (r_ != CURAND_STATUS_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

static void stats(const float *v, size_t n, double *mean, double *var) {
    double s = 0, s2 = 0;
    for (size_t i = 0; i < n; i++) { s += v[i]; s2 += (double)v[i] * v[i]; }
    *mean = s / n;
    *var = s2 / n - *mean * *mean;
}

int main(void) {
    enum { N = 1 << 20 };
    float *dev = dev_alloc(N * sizeof(float)), *host = malloc(N * sizeof(float)), *again = malloc(N * sizeof(float));
    curandGenerator_t gen;
    CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND(curandGenerateUniform(gen, dev, N));
    to_host(host, dev, N * sizeof(float));
    double mean, var;
    stats(host, N, &mean, &var);
    for (int i = 0; i < N; i++) if (!(host[i] > 0.0f && host[i] <= 1.0f)) FAIL("uniform value %g out of (0,1]", host[i]);
    if (fabs(mean - 0.5) > 0.002 || fabs(var - 1.0 / 12) > 0.002) FAIL("uniform mean %g variance %g", mean, var);
    CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND(curandSetGeneratorOffset(gen, 0));
    CURAND(curandGenerateUniform(gen, dev, N));
    to_host(again, dev, N * sizeof(float));
    if (memcmp(host, again, N * sizeof(float))) FAIL("the same seed gave other numbers");
    CURAND(curandGenerateNormal(gen, dev, N, 2.0f, 3.0f));
    to_host(host, dev, N * sizeof(float));
    stats(host, N, &mean, &var);
    if (fabs(mean - 2.0) > 0.02 || fabs(sqrt(var) - 3.0) > 0.02) FAIL("normal mean %g deviation %g", mean, sqrt(var));
    int version = 0;
    CURAND(curandGetVersion(&version));
    CURAND(curandDestroyGenerator(gen));
    printf("cuRAND version reported: %d; uniform mean/variance, normal mean/deviation, and repeatability as expected\n", version);
    OK("curand");
    return 0;
}
