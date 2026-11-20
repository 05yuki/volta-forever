// Shared by the programs of tests/gpu/forward.sh: each is built against the CUDA 13.5 headers and run
// with the forwarding libraries; it calls a few functions of one library and checks the results.
// It prints "ok <library>" for each library it has checked, and exits 1 at the first failure.
#ifndef FORWARD_COMMON_H
#define FORWARD_COMMON_H
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAIL(...) do { fprintf(stderr, "FAILED at line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)
#define CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) FAIL("%s -> %d (%s)", #call, (int)e_, cudaGetErrorString(e_)); } while (0)
#define OK(library) printf("ok %s\n", library)

__attribute__((unused)) static void *dev_alloc(size_t bytes) {
    void *p = 0;
    CUDA(cudaMalloc(&p, bytes));
    return p;
}

__attribute__((unused)) static void to_dev(void *dev, const void *host, size_t bytes) { CUDA(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice)); }
__attribute__((unused)) static void to_host(void *host, const void *dev, size_t bytes) { CUDA(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost)); }

#endif
