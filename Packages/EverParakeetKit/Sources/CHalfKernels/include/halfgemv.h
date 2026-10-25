#ifndef HALFGEMV_H
#define HALFGEMV_H

#include <stdint.h>

/// y[m] = bias[m] + sum_k W[m, k] * x[k] for rows [row_lo, row_hi).
/// W is row-major fp16 (raw IEEE half bits), x / bias / y are fp32, accumulation is fp32.
/// n must be a multiple of 8.
void eph_gemv_f16w(const uint16_t *w, const float *x, const float *bias, float *y,
                   int row_lo, int row_hi, int n);

#endif
