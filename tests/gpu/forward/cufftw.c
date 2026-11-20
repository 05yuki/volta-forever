// cuFFTW: the FFTW-style interface (plan, execute) on host arrays, against a host DFT.
#include "common.h"
#include <cufftw.h>

int main(void) {
    enum { N = 256 };
    fftwf_complex *in = fftwf_malloc(N * sizeof *in), *out = fftwf_malloc(N * sizeof *out);
    for (int i = 0; i < N; i++) {
        in[i][0] = sinf(0.3f * i) + 0.1f * (i % 5);
        in[i][1] = cosf(0.11f * i);
    }
    fftwf_plan plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) FAIL("fftwf_plan_dft_1d returned NULL");
    fftwf_execute(plan);
    double worst = 0;
    for (int k = 0; k < N; k += 5) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * 3.14159265358979323846 * k * n / N;
            re += in[n][0] * cos(a) - in[n][1] * sin(a);
            im += in[n][0] * sin(a) + in[n][1] * cos(a);
        }
        worst = fmax(worst, fmax(fabs(re - out[k][0]), fabs(im - out[k][1])));
    }
    if (worst > 1e-2) FAIL("transform differs from the host DFT by %g", worst);
    fftwf_destroy_plan(plan);
    printf("cuFFTW: max error %g\n", worst);
    OK("cufftw");
    return 0;
}
