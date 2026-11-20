/* Minimal npp.h for the mock library: the forwarder generator reads this with gcc -aux-info. */
#ifndef NPP_H
#define NPP_H

/* Must match the real 13.5 header values so the renumbering table is built. */
typedef enum {
    NPP_SUCCESS = 0,
    NPP_STEP_ERROR = -16,
    NPP_NOT_SUPPORTED_MODE_ERROR = -14,
    NPP_SCALE_RANGE_ERROR = -12,
    NPP_DATA_TYPE_ERROR = -59,
    NPP_ZERO_MASK_VALUE_ERROR = -62,
    NPP_FFT_ORDER_ERROR = -17,
    NPP_FFT_FLAG_ERROR = -18,
    NPP_THRESHOLD_ERROR = -19,
    NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR = -20,
    NPP_MOMENT_00_ZERO_ERROR = -21,
    NPP_MIRROR_FLIP_ERROR = -22,
    NPP_INTERPOLATION_ERROR = -23,
    NPP_RESIZE_FACTOR_ERROR = -24,
    NPP_CONTEXT_MATCH_ERROR = -13,
    NPP_NULL_POINTER_ERROR = -4
} NppStatus;

typedef unsigned char Npp8u;

typedef struct { int width; int height; } NppiSize;
typedef struct { int x; int y; } NppiPoint;
typedef struct { int x; int y; int width; int height; } NppiRect;

/* NppStreamContext: 48 bytes, matches the real one. */
typedef struct {
    int    nCudaDeviceId;
    void  *hStream;
    int    nMultiProcessorCount;
    int    nMaxThreadsPerMultiProcessor;
    int    nMaxThreadsPerBlock;
    int    nSharedMemPerBlock;
    int    nCudaDevAttrComputeCapabilityMajor;
    int    nCudaDevAttrComputeCapabilityMinor;
    unsigned int nStreamFlags;
    int    padding;
} NppStreamContext;

/* Function declarations: must use the real NPP names so the forwarder binds them. */
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int nMaxMarkerLabelID, int *pBufferSize);

NppStatus nppiLabelMarkersUF_32u_C1R(long a, long b, long c, long d, long e, long f, long g);

NppStatus nppiNormDiff_L2_8u_C1R(double d0, double d1, double d2, double d3,
                                  double d4, double d5, double d6, double d7,
                                  double d8);

NppStatus nppiSet_8u_C1R(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI);

NppStatus nppiResizeSqrPixel_8u_C1R(long a, long b, long c, long d, long e, NppiRect roi);

NppStatus nppiAdd_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step,
                                 Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);

NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppiSize oMaskSize,
                                    NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx);

#endif
