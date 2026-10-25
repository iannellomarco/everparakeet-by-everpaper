#include "halfgemv.h"
#include <arm_neon.h>

// fp16 weights widened on the fly to fp32 (exact), fp32 FMA accumulation: the same numbers as a fp32 matrix
// built from the checkpoint's fp16 values, with half the memory traffic.
static inline float32x4_t lo(uint16x8_t h) { return vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))); }
static inline float32x4_t hi(uint16x8_t h) { return vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))); }

void eph_gemv_f16w(const uint16_t *w, const float *x, const float *bias, float *y,
                   int row_lo, int row_hi, int n) {
    int m = row_lo;
    for (; m + 4 <= row_hi; m += 4) {
        const uint16_t *r0 = w + (long)m * n, *r1 = r0 + n, *r2 = r1 + n, *r3 = r2 + n;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (int k = 0; k < n; k += 8) {
            float32x4_t xl = vld1q_f32(x + k), xh = vld1q_f32(x + k + 4);
            uint16x8_t h0 = vld1q_u16(r0 + k), h1 = vld1q_u16(r1 + k), h2 = vld1q_u16(r2 + k), h3 = vld1q_u16(r3 + k);
            a0 = vfmaq_f32(a0, lo(h0), xl); b0 = vfmaq_f32(b0, hi(h0), xh);
            a1 = vfmaq_f32(a1, lo(h1), xl); b1 = vfmaq_f32(b1, hi(h1), xh);
            a2 = vfmaq_f32(a2, lo(h2), xl); b2 = vfmaq_f32(b2, hi(h2), xh);
            a3 = vfmaq_f32(a3, lo(h3), xl); b3 = vfmaq_f32(b3, hi(h3), xh);
        }
        y[m]     = bias[m]     + vaddvq_f32(vaddq_f32(a0, b0));
        y[m + 1] = bias[m + 1] + vaddvq_f32(vaddq_f32(a1, b1));
        y[m + 2] = bias[m + 2] + vaddvq_f32(vaddq_f32(a2, b2));
        y[m + 3] = bias[m + 3] + vaddvq_f32(vaddq_f32(a3, b3));
    }
    for (; m < row_hi; m++) {
        const uint16_t *r0 = w + (long)m * n;
        float32x4_t a0 = vdupq_n_f32(0), b0 = a0;
        for (int k = 0; k < n; k += 8) {
            uint16x8_t h0 = vld1q_u16(r0 + k);
            a0 = vfmaq_f32(a0, lo(h0), vld1q_f32(x + k));
            b0 = vfmaq_f32(b0, hi(h0), vld1q_f32(x + k + 4));
        }
        y[m] = bias[m] + vaddvq_f32(vaddq_f32(a0, b0));
    }
}

