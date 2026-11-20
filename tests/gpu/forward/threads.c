// First calls from many threads at once: every thread waits at a barrier, then calls a function of a
// forwarder that nobody has called yet. The 12.9 library must be loaded once, every call must get its
// answer. (Needs no GPU: version queries only.)
#include <cublas_v2.h>
#include <cufft.h>
#include <curand.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { THREADS = 32 };
static pthread_barrier_t barrier;
static int wrong;

static void *run(void *arg) {
    long id = (long)arg;
    pthread_barrier_wait(&barrier);
    int major = 0, version = 0;
    int bad = 0;
    switch (id % 4) {
    case 0: bad = cublasGetProperty(MAJOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS || major != 12; break;
    case 1: bad = cufftGetVersion(&version) != CUFFT_SUCCESS || version / 1000 != 11; break;
    case 2: bad = curandGetVersion(&version) != CURAND_STATUS_SUCCESS || version < 10000; break;
    default: bad = cublasGetProperty(MINOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS; break;
    }
    if (bad) __atomic_fetch_add(&wrong, 1, __ATOMIC_RELAXED);
    return 0;
}

// Number of mappings of the executable code of a library whose file name contains `name`.
static int mappings(const char *name) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[1024];
    int n = 0;
    while (f && fgets(line, sizeof line, f)) if (strstr(line, name) && strstr(line, " r-xp ")) n++;
    if (f) fclose(f);
    return n;
}

int main(void) {
    pthread_t t[THREADS];
    pthread_barrier_init(&barrier, 0, THREADS);
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], 0, run, (void *)i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], 0);
    int blas = mappings("libcublas.so.12"), fft = mappings("libcufft.so.11"), rnd = mappings("libcurand.so.10.3");
    if (wrong || blas != 1 || fft != 1 || rnd != 1) {
        fprintf(stderr, "FAILED: %d wrong answers; code mappings: cublas %d, cufft %d, curand %d (want 1 each)\n", wrong, blas, fft, rnd);
        return 1;
    }
    printf("ok threads\n");
    return 0;
}
