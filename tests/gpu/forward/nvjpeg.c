// nvJPEG: encodes a smooth RGB image on the GPU (nvjpegEncodeImage) and decodes it again
// (nvjpegDecode), and compares with the original.
#include "common.h"
#include <nvjpeg.h>

#define JPEG(call) do { nvjpegStatus_t r_ = (call); if (r_ != NVJPEG_STATUS_SUCCESS) FAIL("%s -> %d", #call, (int)r_); } while (0)

int main(void) {
    enum { W = 128, H = 96 };
    unsigned char *rgb = malloc(W * H * 3), *back = malloc(W * H * 3);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        unsigned char *p = rgb + 3 * (y * W + x);
        p[0] = (unsigned char)(x * 255 / (W - 1));
        p[1] = (unsigned char)(y * 255 / (H - 1));
        p[2] = (unsigned char)(128 + 100 * sin(x * 0.05) * cos(y * 0.07));
    }
    nvjpegHandle_t h;
    JPEG(nvjpegCreateSimple(&h));
    nvjpegEncoderState_t es;
    nvjpegEncoderParams_t ep;
    JPEG(nvjpegEncoderStateCreate(h, &es, 0));
    JPEG(nvjpegEncoderParamsCreate(h, &ep, 0));
    JPEG(nvjpegEncoderParamsSetQuality(ep, 95, 0));
    JPEG(nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_444, 0));
    nvjpegImage_t img;
    memset(&img, 0, sizeof img);
    img.channel[0] = dev_alloc(W * H * 3);
    img.pitch[0] = W * 3;
    to_dev(img.channel[0], rgb, W * H * 3);
    JPEG(nvjpegEncodeImage(h, es, ep, &img, NVJPEG_INPUT_RGBI, W, H, 0));
    size_t length = 0;
    JPEG(nvjpegEncodeRetrieveBitstream(h, es, NULL, &length, 0));
    unsigned char *jpeg = malloc(length);
    JPEG(nvjpegEncodeRetrieveBitstream(h, es, jpeg, &length, 0));
    CUDA(cudaStreamSynchronize(0));
    if (length < 500 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) FAIL("not a JPEG stream (%zu bytes)", length);

    nvjpegJpegState_t ds;
    JPEG(nvjpegJpegStateCreate(h, &ds));
    int comps, w = 0, hh = 0;
    nvjpegChromaSubsampling_t sub;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    JPEG(nvjpegGetImageInfo(h, jpeg, length, &comps, &sub, widths, heights));
    w = widths[0]; hh = heights[0];
    if (w != W || hh != H) FAIL("decoded size %dx%d", w, hh);
    nvjpegImage_t out;
    memset(&out, 0, sizeof out);
    out.channel[0] = dev_alloc(W * H * 3);
    out.pitch[0] = W * 3;
    JPEG(nvjpegDecode(h, ds, jpeg, length, NVJPEG_OUTPUT_RGBI, &out, 0));
    CUDA(cudaStreamSynchronize(0));
    to_host(back, out.channel[0], W * H * 3);
    double sum = 0;
    int worst = 0;
    for (int i = 0; i < W * H * 3; i++) { int d = abs(back[i] - rgb[i]); sum += d; if (d > worst) worst = d; }
    if (sum / (W * H * 3) > 3.0 || worst > 40) FAIL("decoded image: mean error %g, worst %d", sum / (W * H * 3), worst);
    int major = 0, minor = 0, patch = 0;
    JPEG(nvjpegGetProperty(MAJOR_VERSION, &major));
    JPEG(nvjpegGetProperty(MINOR_VERSION, &minor));
    JPEG(nvjpegGetProperty(PATCH_LEVEL, &patch));
    printf("nvJPEG version reported: %d.%d.%d; %zu-byte JPEG, decoded: mean error %g, worst %d\n", major, minor, patch, length, sum / (W * H * 3), worst);
    JPEG(nvjpegJpegStateDestroy(ds));
    JPEG(nvjpegEncoderParamsDestroy(ep));
    JPEG(nvjpegEncoderStateDestroy(es));
    JPEG(nvjpegDestroy(h));
    OK("nvjpeg");
    return 0;
}
