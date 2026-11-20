/* A stand-in NPP library that the forwarders can be pointed at. Each function records its arguments
   in mock_last and returns NPP_STEP_ERROR_12 (the 12.9 value -14); the forwarder renumbers it to
   NPP_STEP_ERROR_13 (-16). The test checks both the returned status and the recorded arguments.

   Functions cover the cases of the System V x86-64 calling convention:
     case_regs_only        : 2 ints, both in registers, 0 stack words
     case_seven_ints       : 7 ints, 6 in regs + 1 on stack
     case_nine_doubles     : 9 doubles, 8 in SSE regs + 1 on stack
     case_small_struct     : NppiSize (8 B, INTEGER, 1 int reg) + 4 ints, all in regs
     case_rect             : NppiRect (16 B, 2 int regs) + 3 ints, all in regs (5 int)
     case_large_struct     : NppStreamContext (48 B, MEMORY) + 2 ints, ctx on stack
     case_mix              : 5 ints + NppiSize + NppStreamContext, ctx on stack
*/
#include "mock_npp.h"
#include <string.h>

struct mock_record mock_last;

#define VISIBILITY __attribute__((visibility("default")))

/* The mock functions must have the same names as real NPP exports so that the forwarder
   generator finds them and the generated library forwards to them.  We picked function names
   that exist in the real 13.5 headers and whose prototypes match our mock signatures. Actually,
   for the stand-in approach the forwarder generator is pointed at this library, so it only needs
   to export the same names.  But we use real NPP names to stay realistic. */

/* case_regs_only: nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int, int *) */
VISIBILITY NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R(int n, int *pBufSize) {
    mock_last.count = 2;
    mock_last.values[0] = n;
    mock_last.values[1] = (long)(uintptr_t)pBufSize;
    return NPP_STEP_ERROR_12;
}

/* case_seven_ints: nppiLabelMarkersUF_32u_C1R(a..g)  --  the real function takes more args
   but the generator sees the prototype we give it, not the real one.  For this test we build
   the forwarder against the mock headers, so the prototype is ours. */
VISIBILITY NppStatus nppiLabelMarkersUF_32u_C1R(long a, long b, long c, long d, long e, long f, long g) {
    mock_last.count = 7;
    mock_last.values[0] = a; mock_last.values[1] = b; mock_last.values[2] = c;
    mock_last.values[3] = d; mock_last.values[4] = e; mock_last.values[5] = f;
    mock_last.values[6] = g;
    return NPP_STEP_ERROR_12;
}

/* case_nine_doubles: nppiNormDiff_L2_8u_C1R(d0..d8)  -- mock signature */
VISIBILITY NppStatus nppiNormDiff_L2_8u_C1R(double d0, double d1, double d2, double d3,
                                             double d4, double d5, double d6, double d7,
                                             double d8) {
    mock_last.count = 9;
    long *v = mock_last.values;
    memcpy(&v[0], &d0, 8); memcpy(&v[1], &d1, 8); memcpy(&v[2], &d2, 8);
    memcpy(&v[3], &d3, 8); memcpy(&v[4], &d4, 8); memcpy(&v[5], &d5, 8);
    memcpy(&v[6], &d6, 8); memcpy(&v[7], &d7, 8); memcpy(&v[8], &d8, 8);
    return NPP_STEP_ERROR_12;
}

/* case_small_struct: nppiSet_8u_C1R(Npp8u value, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
   -- the real function; NppiSize (8B) fits in 1 int reg, 4 params total = 4 int regs. */
VISIBILITY NppStatus nppiSet_8u_C1R(unsigned char val, unsigned char *pDst, int step, NppiSize roi) {
    mock_last.count = 4;
    mock_last.values[0] = val;
    mock_last.values[1] = (long)(uintptr_t)pDst;
    mock_last.values[2] = step;
    mock_last.values[3] = (long)roi.width << 32 | (long)(unsigned)roi.height;
    return NPP_STEP_ERROR_12;
}

/* case_rect: nppiMirror_8u_C1R(src, srcStep, dst, dstStep, NppiRect roi)
   NppiRect is 16 bytes, 2 int regs.  5 int params + 2 from rect = 7 > 6, so rect on stack?
   No: 4 ptr/int params use 4 int regs, leaving 2 for rect.  Wait, let me count:
   src(ptr)=rdi, srcStep=rsi, dst(ptr)=rdx, dstStep=rcx, roi(2 int regs)=r8+r9. Total 6. All fit.
   Let me use a mock with 5 ints + NppiRect = 7 int regs needed, so rect goes on stack. */
VISIBILITY NppStatus nppiResizeSqrPixel_8u_C1R(long a, long b, long c, long d, long e, NppiRect roi) {
    mock_last.count = 6;
    mock_last.values[0] = a; mock_last.values[1] = b; mock_last.values[2] = c;
    mock_last.values[3] = d; mock_last.values[4] = e;
    mock_last.values[5] = (long)roi.x << 48 | (long)(unsigned short)roi.y << 32
                         | (long)(unsigned short)roi.width << 16 | (unsigned short)roi.height;
    return NPP_STEP_ERROR_12;
}

/* case_large_struct: nppiAdd_8u_C1RSfs_Ctx(src1, s1step, src2, s2step, dst, dstep, roi, scale, ctx)
   NppStreamContext (48B) is MEMORY class: always on the stack (6 words).
   The first 8 params (6 ptr/int + NppiSize + int) use: 6 int regs for the first 6, NppiSize on stack
   because regs exhausted... wait, NppiSize is 1 int reg: 6 ptr/int used 6, so NppiSize goes on stack.
   Actually let me count: src1(ptr), s1step(int), src2(ptr), s2step(int), dst(ptr), dstep(int) = 6 int regs.
   roi(NppiSize, 1 int reg): no regs left, on stack (1 word). scale(int): on stack (1 word).
   ctx(NppStreamContext, MEMORY): on stack (6 words). Total: 8 stack words. */
VISIBILITY NppStatus nppiAdd_8u_C1RSfs_Ctx(const void *s1, int ss1, const void *s2, int ss2,
                                            void *dst, int dstep, NppiSize roi, int scale,
                                            NppStreamContext ctx) {
    mock_last.count = 9;
    mock_last.values[0] = (long)(uintptr_t)s1;
    mock_last.values[1] = ss1;
    mock_last.values[2] = (long)(uintptr_t)s2;
    mock_last.values[3] = ss2;
    mock_last.values[4] = (long)(uintptr_t)dst;
    mock_last.values[5] = dstep;
    mock_last.values[6] = roi.width;
    mock_last.values[7] = scale;
    mock_last.values[8] = ctx.nCudaDeviceId;
    return NPP_STEP_ERROR_12;
}

/* case_mix: nppiFilterBox_8u_C1R_Ctx(src, srcStep, dst, dstStep, NppiSize roi, NppiSize mask,
                                       NppiPoint anchor, NppStreamContext ctx)
   8 int params in 6 regs (src, srcStep, dst, dstStep, roi, mask): roi(1 int reg) and mask(1 int reg)
   use the 5th and 6th int regs.  anchor(NppiPoint, 1 int reg) goes on stack.
   ctx(NppStreamContext, MEMORY) goes on stack.  Total stack: 1 + 6 = 7 → 8 (rounded to even). */
VISIBILITY NppStatus nppiFilterBox_8u_C1R_Ctx(const void *src, int srcStep, void *dst, int dstStep,
                                               NppiSize roi, NppiSize mask, NppiPoint anchor,
                                               NppStreamContext ctx) {
    mock_last.count = 8;
    mock_last.values[0] = (long)(uintptr_t)src;
    mock_last.values[1] = srcStep;
    mock_last.values[2] = (long)(uintptr_t)dst;
    mock_last.values[3] = dstStep;
    mock_last.values[4] = roi.width;
    mock_last.values[5] = mask.width;
    mock_last.values[6] = anchor.x;
    mock_last.values[7] = ctx.nCudaDeviceId;
    return NPP_STEP_ERROR_12;
}
