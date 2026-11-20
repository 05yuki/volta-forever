/* The "12.9" header: same declarations, but NppStatus enumerators have the old (12.9) values. */
#ifndef NPP_H
#define NPP_H

typedef enum {
    NPP_SUCCESS = 0,
    NPP_STEP_ERROR = -14,
    NPP_NOT_SUPPORTED_MODE_ERROR = -9999,
    NPP_SCALE_RANGE_ERROR = -13,
    NPP_DATA_TYPE_ERROR = -12,
    NPP_ZERO_MASK_VALUE_ERROR = -59,
    NPP_FFT_ORDER_ERROR = -15,
    NPP_FFT_FLAG_ERROR = -16,
    NPP_THRESHOLD_ERROR = -18,
    NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR = -19,
    NPP_MOMENT_00_ZERO_ERROR = -20,
    NPP_MIRROR_FLIP_ERROR = -21,
    NPP_INTERPOLATION_ERROR = -22,
    NPP_RESIZE_FACTOR_ERROR = -23,
    NPP_CONTEXT_MATCH_ERROR = -17,
    NPP_NULL_POINTER_ERROR = -4
} NppStatus;

typedef unsigned char Npp8u;

typedef struct { int width; int height; } NppiSize;
typedef struct { int x; int y; } NppiPoint;
typedef struct { int x; int y; int width; int height; } NppiRect;

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
