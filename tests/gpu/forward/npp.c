// NPP: one function from each library (nppc version, nppisu allocation, nppidei set and copy, nppial
// add, nppif box filter, nppim erosion, nppitc threshold, nppicc colour conversion, nppig resize,
// nppist sum, npps vector add), checked against the host; then the status codes: an error that CUDA
// 12.9's NPP numbers differently from NPP 13.5 must come back with 13.5's number.
#include "common.h"
#include <npp.h>

#define NPP(call) do { NppStatus r_ = (call); if (r_ != NPP_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

enum { W = 40, H = 24 };

static NppStreamContext context(void) {
    NppStreamContext c;
    memset(&c, 0, sizeof c);
    int shared = 0;
    CUDA(cudaGetDevice(&c.nCudaDeviceId));
    c.hStream = 0;
    CUDA(cudaDeviceGetAttribute(&c.nMultiProcessorCount, cudaDevAttrMultiProcessorCount, c.nCudaDeviceId));
    CUDA(cudaDeviceGetAttribute(&c.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, c.nCudaDeviceId));
    CUDA(cudaDeviceGetAttribute(&c.nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, c.nCudaDeviceId));
    CUDA(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlock, c.nCudaDeviceId));
    c.nSharedMemPerBlock = shared;
    CUDA(cudaDeviceGetAttribute(&c.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, c.nCudaDeviceId));
    CUDA(cudaDeviceGetAttribute(&c.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, c.nCudaDeviceId));
    return c;
}

static Npp8u *upload(const Npp8u *host, int channels, int *step) {
    Npp8u *d = nppiMalloc_8u_C1(W * channels, H, step);
    if (!d) FAIL("nppiMalloc_8u_C1 returned NULL");
    CUDA(cudaMemcpy2D(d, *step, host, W * channels, W * channels, H, cudaMemcpyHostToDevice));
    return d;
}

static void download(Npp8u *host, const Npp8u *dev, int step, int width, int height) {
    CUDA(cudaMemcpy2D(host, width, dev, step, width, height, cudaMemcpyDeviceToHost));
}

int main(void) {
    NppStreamContext ctx = context();
    NppiSize roi = {W, H};
    Npp8u a[W * H], b[W * H], rgb[W * H * 3], out[W * H * 3];
    for (int i = 0; i < W * H; i++) { a[i] = (Npp8u)((i * 37) % 251); b[i] = (Npp8u)((i * 11 + 5) % 97); }
    for (int i = 0; i < W * H * 3; i++) rgb[i] = (Npp8u)((i * 29) % 256);

    const NppLibraryVersion *v = nppGetLibVersion();
    if (!v || v->major < 12) FAIL("nppGetLibVersion gave %p", (const void *)v);
    printf("NPP version reported: %d.%d.%d\n", v->major, v->minor, v->build);
    OK("nppc");

    int sa, sb, sd;
    Npp8u *da = upload(a, 1, &sa), *db = upload(b, 1, &sb), *dd = nppiMalloc_8u_C1(W, H, &sd);
    OK("nppisu");

    NPP(nppiSet_8u_C1R_Ctx(77, dd, sd, roi, ctx));
    download(out, dd, sd, W, H);
    for (int i = 0; i < W * H; i++) if (out[i] != 77) FAIL("nppiSet: pixel %d is %d", i, out[i]);
    NPP(nppiCopy_8u_C1R_Ctx(da, sa, dd, sd, roi, ctx));
    download(out, dd, sd, W, H);
    if (memcmp(out, a, W * H)) FAIL("nppiCopy differs");
    OK("nppidei");

    NPP(nppiAdd_8u_C1RSfs_Ctx(da, sa, db, sb, dd, sd, roi, 0, ctx));
    download(out, dd, sd, W, H);
    for (int i = 0; i < W * H; i++) { int s = a[i] + b[i]; if (out[i] != (s > 255 ? 255 : s)) FAIL("nppiAdd: pixel %d is %d, want %d", i, out[i], s); }
    OK("nppial");

    NppiSize mask = {3, 3};
    NppiPoint anchor = {1, 1};
    NppiSize inner = {W - 2, H - 2};   // the filter reads the neighbours: stay inside the image
    NPP(nppiSet_8u_C1R_Ctx(0, dd, sd, roi, ctx));
    NPP(nppiFilterBox_8u_C1R_Ctx(da + sa + 1, sa, dd + sd + 1, sd, inner, mask, anchor, ctx));
    download(out, dd, sd, W, H);
    for (int y = 1; y < H - 1; y++) for (int x = 1; x < W - 1; x++) {
        int s = 0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) s += a[(y + j) * W + x + i];
        int want = (s + 4) / 9, got = out[y * W + x];
        if (abs(got - want) > 1) FAIL("nppiFilterBox: pixel (%d,%d) is %d, want %d", x, y, got, want);
    }
    OK("nppif");

    NPP(nppiSet_8u_C1R_Ctx(0, dd, sd, roi, ctx));
    NPP(nppiErode3x3_8u_C1R_Ctx(da + sa + 1, sa, dd + sd + 1, sd, inner, ctx));
    download(out, dd, sd, W, H);
    for (int y = 1; y < H - 1; y++) for (int x = 1; x < W - 1; x++) {
        int m = 255;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) m = a[(y + j) * W + x + i] < m ? a[(y + j) * W + x + i] : m;
        if (out[y * W + x] != m) FAIL("nppiErode3x3: pixel (%d,%d) is %d, want %d", x, y, out[y * W + x], m);
    }
    OK("nppim");

    NPP(nppiThreshold_8u_C1R_Ctx(da, sa, dd, sd, roi, 100, NPP_CMP_GREATER, ctx));  // pixels above 100 become 100
    download(out, dd, sd, W, H);
    for (int i = 0; i < W * H; i++) { int want = a[i] > 100 ? 100 : a[i]; if (out[i] != want) FAIL("nppiThreshold: pixel %d is %d, want %d", i, out[i], want); }
    OK("nppitc");

    int s3;
    Npp8u *d3 = upload(rgb, 3, &s3);
    NPP(nppiRGBToGray_8u_C3C1R_Ctx(d3, s3, dd, sd, roi, ctx));
    download(out, dd, sd, W, H);
    for (int i = 0; i < W * H; i++) {
        double want = 0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2];
        if (fabs(out[i] - want) > 1.01) FAIL("nppiRGBToGray: pixel %d is %d, want %.1f", i, out[i], want);
    }
    OK("nppicc");

    NppiRect whole = {0, 0, W, H}, half = {0, 0, W / 2, H / 2};
    int sh;
    Npp8u *dh = nppiMalloc_8u_C1(W / 2, H / 2, &sh);
    NPP(nppiResize_8u_C1R_Ctx(da, sa, roi, whole, dh, sh, (NppiSize){W / 2, H / 2}, half, NPPI_INTER_NN, ctx));
    download(out, dh, sh, W / 2, H / 2);
    int nonzero = 0;
    for (int i = 0; i < W / 2 * H / 2; i++) nonzero += out[i] != 0;
    if (nonzero < W * H / 8) FAIL("nppiResize produced %d non-zero pixels", nonzero);
    for (int y = 0; y < H / 2; y++) for (int x = 0; x < W / 2; x++) {
        int sx = x * 2, sy = y * 2, got = out[y * (W / 2) + x];
        if (got != a[sy * W + sx] && got != a[(sy + 1) * W + sx] && got != a[sy * W + sx + 1] && got != a[(sy + 1) * W + sx + 1])
            FAIL("nppiResize: pixel (%d,%d) is %d, none of its source pixels", x, y, got);
    }
    OK("nppig");

    size_t bytes = 0;
    NPP(nppiSumGetBufferHostSize_8u_C1R_Ctx(roi, &bytes, ctx));
    Npp8u *buf = dev_alloc(bytes ? bytes : 1);
    Npp64f *dsum = dev_alloc(sizeof(Npp64f));
    NPP(nppiSum_8u_C1R_Ctx(da, sa, roi, buf, dsum, ctx));
    Npp64f sum = 0, want_sum = 0;
    to_host(&sum, dsum, sizeof sum);
    for (int i = 0; i < W * H; i++) want_sum += a[i];
    if (sum != want_sum) FAIL("nppiSum is %.0f, want %.0f", sum, want_sum);
    OK("nppist");

    enum { L = 1000 };
    Npp32f *x = nppsMalloc_32f(L), *y = nppsMalloc_32f(L), *z = nppsMalloc_32f(L);
    float hx[L], hy[L], hz[L];
    for (int i = 0; i < L; i++) { hx[i] = i * 0.5f; hy[i] = 100.0f - i; }
    to_dev(x, hx, sizeof hx); to_dev(y, hy, sizeof hy);
    NPP(nppsAdd_32f_Ctx(x, y, z, L, ctx));
    to_host(hz, z, sizeof hz);
    for (int i = 0; i < L; i++) if (hz[i] != hx[i] + hy[i]) FAIL("nppsAdd: element %d is %g", i, hz[i]);
    nppsFree(x); nppsFree(y); nppsFree(z);
    OK("npps");

    // Status numbering: three errors that CUDA 12.9's NPP numbers differently from NPP 13.5 (a bad step
    // -14 -> -16, a bad interpolation mode -22 -> -23, an unsupported mode -9999 -> -14) must come back
    // with 13.5's numbers, those of the header this program was built with.
    NppStatus st = nppiAdd_8u_C1RSfs_Ctx(da, sa, db, sb, dd, 0, roi, 0, ctx);
    if (st != NPP_STEP_ERROR) FAIL("a zero destination step returned %d, NPP_STEP_ERROR is %d", (int)st, (int)NPP_STEP_ERROR);
    NppiRect whole2 = {0, 0, W, H};
    st = nppiResize_8u_C1R_Ctx(da, sa, roi, whole2, dd, sd, roi, whole2, 12345, ctx);
    if (st != NPP_INTERPOLATION_ERROR) FAIL("a bad interpolation mode returned %d, NPP_INTERPOLATION_ERROR is %d", (int)st, (int)NPP_INTERPOLATION_ERROR);
    st = nppiThreshold_8u_C1R_Ctx(da, sa, dd, sd, roi, 5, (NppCmpOp)77, ctx);
    if (st != NPP_NOT_SUPPORTED_MODE_ERROR) FAIL("a bad comparison returned %d, NPP_NOT_SUPPORTED_MODE_ERROR is %d", (int)st, (int)NPP_NOT_SUPPORTED_MODE_ERROR);
    st = nppiAdd_8u_C1RSfs_Ctx(NULL, sa, db, sb, dd, sd, roi, 0, ctx);
    if (st != NPP_NULL_POINTER_ERROR) FAIL("a null pointer returned %d, want %d", (int)st, (int)NPP_NULL_POINTER_ERROR);
    printf("statuses: NPP_STEP_ERROR %d, NPP_INTERPOLATION_ERROR %d, NPP_NOT_SUPPORTED_MODE_ERROR %d, NPP_NULL_POINTER_ERROR %d (13.5's numbers)\n",
           (int)NPP_STEP_ERROR, (int)NPP_INTERPOLATION_ERROR, (int)NPP_NOT_SUPPORTED_MODE_ERROR, (int)NPP_NULL_POINTER_ERROR);
    printf("ok npp-status\n");
    return 0;
}
