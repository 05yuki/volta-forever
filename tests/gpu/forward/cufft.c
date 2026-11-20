// cuFFT: a 1-D complex transform of 1024 points on the GPU against a host DFT, and the inverse.
#include "common.h"
#include <cufft.h>

#define CUFFT(call) do { cufftResult r_ = (call); if (r_ != CUFFT_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

int main(void) {
    enum { N = 1024 };
    cufftComplex *in = malloc(N * sizeof *in), *out = malloc(N * sizeof *out);
    for (int i = 0; i < N; i++) {
        in[i].x = sinf(2.0f * 3.14159265f * 5.0f * i / N) + 0.25f * cosf(2.0f * 3.14159265f * 37.0f * i / N);
        in[i].y = 0.5f * (float)((i * 7) % 13) / 13.0f;
    }
    cufftComplex *dev = dev_alloc(N * sizeof *dev);
    to_dev(dev, in, N * sizeof *dev);
    cufftHandle plan;
    CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CUFFT(cufftExecC2C(plan, dev, dev, CUFFT_FORWARD));
    to_host(out, dev, N * sizeof *dev);
    double worst = 0;
    for (int k = 0; k < N; k += 7) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * 3.14159265358979323846 * k * n / N;
            re += in[n].x * cos(a) - in[n].y * sin(a);
            im += in[n].x * sin(a) + in[n].y * cos(a);
        }
        worst = fmax(worst, fmax(fabs(re - out[k].x), fabs(im - out[k].y)));
    }
    if (worst > 1e-2) FAIL("forward transform differs from the host DFT by %g", worst);
    CUFFT(cufftExecC2C(plan, dev, dev, CUFFT_INVERSE));
    to_host(out, dev, N * sizeof *dev);
    double back = 0;
    for (int i = 0; i < N; i++) back = fmax(back, fmax(fabs(out[i].x / N - in[i].x), fabs(out[i].y / N - in[i].y)));
    if (back > 1e-4) FAIL("inverse transform is off by %g", back);
    int version = 0;
    CUFFT(cufftGetVersion(&version));
    CUFFT(cufftDestroy(plan));
    printf("cuFFT version reported: %d; forward vs host DFT: max error %g; round trip: %g\n", version, worst, back);
    OK("cufft");
    return 0;
}
