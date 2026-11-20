/* Types that mirror the NPP types used by the test, without depending on NVIDIA's headers.
   The forwarder generator reads the real headers; this file is only for the mock library
   and the test harness. */
#ifndef MOCK_NPP_H
#define MOCK_NPP_H
#include <stdint.h>

/* 12.9 values: the mock library returns these; the forwarder renumbers them to 13.5's values. */
#define NPP_STEP_ERROR_12     (-14)
#define NPP_STEP_ERROR_13     (-16)
#define NPP_SUCCESS_VAL        0

typedef int NppStatus;

typedef struct { int width; int height; } NppiSize;
typedef struct { int x; int y; } NppiPoint;
typedef struct { int x; int y; int width; int height; } NppiRect;

/* 48 bytes, > 16: MEMORY class, always passed on the stack. */
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

/* Arguments recorded by the mock library so the test can verify them. */
struct mock_record {
    long values[32];
    int  count;
};
extern struct mock_record mock_last;

#endif
